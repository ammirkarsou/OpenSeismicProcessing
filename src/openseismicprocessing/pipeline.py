import inspect

def run_pipeline(steps):
    context = {}

    for i, (func, kwargs) in enumerate(steps):
        output_key = kwargs.pop("output", None)
        func_name = func.__name__

        # Resolve @context references
        resolved_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value.startswith("@"):
                ref_key = value[1:]
                if ref_key in context:
                    resolved_kwargs[key] = context[ref_key]
                else:
                    print(f"❌ Error: Referenced key '{ref_key}' not found in context for step {i+1} ({func_name})")
                    return context
            else:
                resolved_kwargs[key] = value

        # Detect if function expects context injection
        inject_context = "context" in inspect.signature(func).parameters

        try:
            result = func(context, **resolved_kwargs) if inject_context else func(**resolved_kwargs)

            if result is None and output_key:
                print(f"❌ Error: Step {i+1} ({func_name}) returned None while expecting output '{output_key}'. Pipeline halted.")
                return context

            if output_key and result is not None:
                context[output_key] = result
            elif output_key and result is None:
                # Already handled above, just being explicit
                pass
            else:
                # No output key → side-effect function, it's fine if it returns None
                pass

        except Exception as e:
            print(f"❌ Exception in step {i+1} ({func_name}): {e}")
            context[f"{func_name}_error"] = str(e)
            break

    return context


def print_pipeline_steps(steps):
    print("📋 Processing Pipeline\n" + "-"*30)
    for i, (func, kwargs) in enumerate(steps):
        name = func.__name__
        output = kwargs.get("output", "—")
        inputs = [f"{k}={v}" for k, v in kwargs.items() if k != "output"]
        print(f"{i+1:02d}. {name}")
        print(f"    Inputs : {', '.join(inputs)}")
        print(f"    Output : {output}\n")