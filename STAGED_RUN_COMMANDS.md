# Staged 1K run controls

Run these from PowerShell in `C:\Users\hasto\EduLLM1`.

## Main run / resume

Use the same command to start and to resume:

```powershell
.\run_staged_1k.ps1 -WithOpus
```

Completed stages are skipped. Do not add `-Force` when resuming.

## Status

```powershell 
.\run_staged_1k.ps1 -Stage status
```

You can also ask Cursor to check the status, logs, ETA, GPU utilization, or
current checkpoint while the run continues.

## Safe pause

Pause after the next complete training checkpoint:

```powershell
.\run_staged_1k.ps1 -Stage pause-next
```

Pause at the first checkpoint after a relative delay:

```powershell
.\run_staged_1k.ps1 -Stage pause-next -AfterMinutes 60
```

The training process checks the request while running and exits only after the
checkpoint has been written safely. Data preparation uses download/year caches
rather than Trainer checkpoints.

## Roll back one checkpoint

First request a pause and wait until status reports `paused`. Then:

```powershell
.\run_staged_1k.ps1 -Stage rollback
```

Rollback requires two retained checkpoints. It archives the newest checkpoint
instead of deleting it, so the preceding checkpoint becomes the next resume
point. Resume with the main command.

## Immediate stop

Press `Ctrl+C` once in the run terminal. This is safe for completed caches but
training work since the most recent checkpoint may need to be repeated.
