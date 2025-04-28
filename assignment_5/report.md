# Assignment 5: Pipeline Parallelism Report

## Q1: Seeding in Different Parallelism Dimensions

1. In pipeline parallelism, seeds across the pipeline parallel dimension should ideally be different. Since each rank manages a different set of layers, using different seeds ensures that each rank initializes its parameters independently. However, if each rank only selects its own portion after initialization, the same seed could work, but different seeds are generally preferable for true parameter independence.

2. In data parallelism, seeds must be the same across all ranks. This is crucial because data parallelism replicates the entire model, and all replicas need identical initialization to properly average gradients during training.

3. In tensor parallelism, it gets more nuanced. Sharded components (linear layers) can use different seeds since they represent different parameter subspaces. However, replicated components (like LayerNorm) must use identical seeds to maintain consistency across ranks.

4. When combining all three parallelism techniques, we face conflicting seed requirements. A hierarchical seed system provides the solution:
   ```python
   seed_full = hash((global_seed, dp_rank, tp_rank, pp_rank))  # For uniqueness
   seed_shared = hash((global_seed, dp_rank, tp_rank))         # For replicated components
   ```
   This ensures parameters are consistent where needed while maintaining diversity where required.

## Q2: Argument Assertions for Balanced Pipeline & Micro-batches

```python
# Assert arguments for balanced pipeline and efficient micro-batches
assert number_of_layers % pp == 0, (
    f"Number of layers ({number_of_layers}) must be divisible by pp ({pp})"
)
assert global_batch_size % micro_batch_size == 0, (
    f"Global batch size ({global_batch_size}) must be divisible by micro batch size ({micro_batch_size})"
)
num_micro_batches = global_batch_size // micro_batch_size
assert num_micro_batches >= 2 * pp, (
    f"Number of microbatches ({num_micro_batches}) must be >= 2 * pp ({2*pp}) "
    "for good pipeline utilization"
)
```

These assertions ensure that layers divide evenly among pipeline stages and that we have enough micro-batches to keep the pipeline filled during execution.

## Q3: Shape of Communicated Tensors

Given our model architecture, the communicated tensors will have shape:
```
[micro_batch_size, sequence_length, hidden_size]
```

With the default values:
- micro_batch_size = 2
- sequence_length = 8192
- hidden_size = 4096

The activation shape being communicated is [2, 8192, 4096]. Since all layers in our model preserve this shape, it remains constant throughout the pipeline.

## Q4: Baseline Forward & Backward (Non-Pipelined)

```python
# Baseline implementation (non-pipelined)
output_tensors_no_pp = []
model = MyDummyModel(number_of_layers, hidden_size, intermediate_size).cuda()
for _ in range(number_of_microbatches):
    # 1. Fetch a microbatch
    micro_batch = next(train_dl_iterator)
    # 2. Move to device
    micro_batch = micro_batch.to(device_id)
    # 3. Compute forward pass
    output = model(micro_batch)
    output_tensors_no_pp.append(output.detach().clone())
    # 4. Compute backward pass with fake loss
    loss = output.mean()
    loss.backward()
```

## Q5: Layer Distribution for Pipeline Stages

```python
def distribute_layers(num_layers: int, pp_rank: int, pp_world_size: int) -> List[int]:
    layers_per_stage = num_layers // pp_world_size
    start = pp_rank * layers_per_stage
    end = start + layers_per_stage
    return list(range(start, end))
```

This function divides the model layers among pipeline stages. For example, with 12 layers and 4 pipeline stages, each stage gets 3 consecutive layers.

## Q6: Which Ranks Use the Dataloader?

Only the first pipeline stage (rank 0) requires the dataloader. All subsequent stages receive their inputs as activations from the previous stage.

```python
if device_mesh["pp"].get_local_rank() == 0:
    train_dl_iterator = iter(input)
else:
    train_dl_iterator = None
```

## Q7: Pipeline Communication Conditions

The pipeline_communicate function handles four distinct operations with specific conditions:

- **recv_forward**: Receives activations from the previous rank during forward pass
  - Condition: All ranks except the first one
  - Communication: pp_rank receives from pp_rank-1

- **send_forward**: Sends activations to the next rank during forward pass
  - Condition: All ranks except the last one
  - Communication: pp_rank sends to pp_rank+1

- **recv_backward**: Receives gradients from the next rank during backward pass
  - Condition: All ranks except the last one
  - Communication: pp_rank receives from pp_rank+1

- **send_backward**: Sends gradients to the previous rank during backward pass
  - Condition: All ranks except the first one
  - Communication: pp_rank sends to pp_rank-1

This communication pattern enables the All-Forward-All-Backward schedule where all microbatches first complete their forward passes before any backward passes begin.

## Q8: Pipelined Forward & Backward

```python
# Pipelined implementation
input_tensors, output_tensors, output_tensors_pp = [], [], []

# All forward passes first
for _ in range(number_of_microbatches):
    # 1. Receive input from previous stage or None
    input_tensor = pipeline_communicate(
        operation='recv_forward',
        pp_process_group=device_mesh["pp"].get_group(),
        shapes=(micro_batch_size, sequence_length, hidden_size)
    )
    
    # 2. Get input from dataloader or from previous stage
    if device_mesh["pp"].get_local_rank() == 0:
        micro_batch = next(train_dl_iterator).to(device_id)
        input_to_model = micro_batch
    else:
        input_to_model = input_tensor
    
    # 3. Forward pass
    output = model_stage(input_to_model)
    
    # 4. Send output to next stage
    pipeline_communicate(
        operation='send_forward',
        pp_process_group=device_mesh["pp"].get_group(),
        tensor=output
    )
    
    output_tensors_pp.append(output.detach().clone())
    input_tensors.append(input_to_model)
    output_tensors.append(output)
    
    # Last stage starts backward pass immediately
    if device_mesh["pp"].get_local_rank() == pp - 1:
        loss = output.mean()
        loss.backward()

# Then all backward passes
for _ in range(number_of_microbatches):
    # Receive gradients from next stage
    output_grad = pipeline_communicate(
        operation='recv_backward',
        pp_process_group=device_mesh["pp"].get_group(),
        shapes=(micro_batch_size, sequence_length, hidden_size)
    )
    
    # Get stored tensors for this microbatch
    inp, out = input_tensors.pop(0), output_tensors.pop(0)
    
    # Backward pass
    inp_grad = model_stage.backward(inp, out, output_grad)
    
    # Send gradients to previous stage
    pipeline_communicate(
        operation='send_backward',
        pp_process_group=device_mesh["pp"].get_group(),
        tensor=inp_grad
    )
```

## Q9: Verifying Outputs

```python
# Only verify outputs on the last pipeline stage
if device_mesh["pp"].get_local_rank() == pp - 1:
    for out_ref, out_pp in zip(output_tensors_no_pp, output_tensors_pp):
        torch.testing.assert_close(
            out_ref, out_pp,
            rtol=1e-2, atol=1e-3
        )
```

We only check outputs on the last pipeline stage since that's where the final model outputs are produced.

## Q10: Verifying Gradients

```python
# Compare gradients for each layer in this pipeline stage
for pp_layer in model_stage.pp_stage_layers:
    idx = pp_layer.layer_idx
    ref_layer = model.layers[idx]
    
    torch.testing.assert_close(
        pp_layer.fc1.weight.grad,
        ref_layer.fc1.weight.grad,
        rtol=1e-2, atol=1e-3
    )
    
    torch.testing.assert_close(
        pp_layer.fc2.weight.grad,
        ref_layer.fc2.weight.grad,
        rtol=1e-2, atol=1e-3
    )
```

Each rank verifies gradients only for its own layers, comparing them with the corresponding layers in the non-pipelined model to ensure numerical equivalence.