import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import time

def maybe_compile(enabled: bool):
    def wrapper(fn):
        return torch.compile(fn) if enabled else fn
    return wrapper

def build_model(use_compile):
    @maybe_compile(enabled=use_compile)
    def toy_example(a, b):
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = b * -1
        return x * b
    return toy_example

def run_loop(model, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning {args.steps} iterations on {device}...")

    start_time = time.time()

    if args.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_logs")
        ) as prof:
            for _ in range(args.steps):
                a = torch.randn(10, device=device)
                b = torch.randn(10, device=device)
                with record_function("toy_example_run"):
                    model(a, b)
                prof.step()

        if args.verbose:
            print("\n📊 Torch Profiler Summary:")
            print(prof.key_averages(group_by_stack_n=args.stack_depth).table(
                sort_by="self_cuda_time_total",
                row_limit=10
            ))
    else:
        for _ in range(args.steps):
            a = torch.randn(10, device=device)
            b = torch.randn(10, device=device)
            model(a, b)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / args.steps

    print(f"✅ Total time: {total_time_ms:.2f} ms")
    print(f"✅ Average latency per call: {avg_latency_ms:.4f} ms\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--depyf", action="store_true", help="Enable depyf debug dumping")
    parser.add_argument("--steps", type=int, default=100, help="Number of iterations to run")
    parser.add_argument("--verbose", action="store_true", help="Print profiler summary")
    parser.add_argument("--stack-depth", type=int, default=0,
                    help="Group profiler ops by N levels of call stack (default: 0 = disabled)")
    args = parser.parse_args()

    model = build_model(use_compile=args.compile)

    if args.depyf:
        import depyf
        with depyf.prepare_debug("depyf_debug_dir"):
            run_loop(model, args)
    else:
        run_loop(model, args)

if __name__ == "__main__":
    main()
