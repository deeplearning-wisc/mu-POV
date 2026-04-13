
def load_data(args, split):
    """Dispatch to the appropriate dataset loader based on args.data."""
    if args.data == 'arithmetics':
        from data.arithmetics import load_data as _load
        return _load(args, split=split)
    elif args.data == 'easy_arithmetics':
        from data.arithmetics import load_data as _load
        return _load(args, split=split, easy=True)
    elif args.data == 'gsm8k':
        from data.gsm8k import load_data as _load
        return _load(args, split=split)
    elif args.data == 'gpqa':
        from data.gpqa import load_data as _load
        return _load(args, split=split)
    elif args.data == 'cnn_daily':
        from data.cnn_daily import load_data as _load
        return _load(args, split=split)
    elif args.data == 'math500':
        from data.math500 import load_data as _load
        return _load(args, split=split)
    elif args.data == 'humaneval':
        from data.humaneval import load_data as _load
        return _load(args, split=split)
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
