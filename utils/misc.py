
def setup_experiment(args, overrides):
    # Load config
    cfg_path = Path(__file__).parent.parent / "configs" / f"{args.config}.yaml"
    cfg = OmegaConf.load(cfg_path)

    # Apply CLI overrides
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # Auto-generate run name if not set
    name = args.name or f"{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Save directory
    save_dir = "runs" / Path(args.save_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    OmegaConf.save(cfg, save_dir / "config.yaml")

    return cfg, save_dir
