import argparse
import json
import os
from core.execute_porosity_model import execute_porosity_model

def parse_params(param_file):
    """Parse the JSON parameter file."""
    with open(param_file, 'r') as f:
        return json.load(f)

def run_model_cli():
    parser = argparse.ArgumentParser(description="Run porosity model (CLI or GUI)")

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch the graphical user interface (GUI) instead of running from CLI.'
    )
    parser.add_argument('--itemids', type=int, nargs='+', default=[28152209, 26304175],
                        help='List of item IDs for dataset download/merge')
    parser.add_argument('--raw_data_dir', type=str, default='rawdata', help='Raw data directory')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--pore_models', type=str, nargs='+',
                        default=['lof', 'keyhole', 'gas'], help='Pore model names')
    parser.add_argument('--meltpool_model', type=str, default='Rosenthal', help='Melt pool model name')
    parser.add_argument('--params', type=str, required=False,
                        help='JSON file containing model parameters (required for CLI mode)')
    parser.add_argument(
        '--totalpores',
        type=int,
        default=None,
        help='Pores to resample. Omit to use every pore on the matched coupons.',
    )
    parser.add_argument('--props', type=str,
                        default='material_models/test_ti64_rosenthal.json',
                        help='Material property dataset file path')
    parser.add_argument('--data_dir', type=str, default='data', help='Merged pore data directory')

    args = parser.parse_args()

    # ---------------- GUI Mode ----------------
    if args.gui:
        print("Launching Porosity Model GUI...")
        from gui.main_window import launch_gui

        launch_gui()
        return

    # ---------------- CLI Mode ----------------
    if not args.params:
        raise ValueError("Missing required argument '--params' when running in CLI mode.")

    param_dict = parse_params(args.params)

    output_dir = args.output_dir
    if not output_dir:
        import datetime
        output_dir = f'output/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    os.makedirs(output_dir, exist_ok=True)

    try:
        results = execute_porosity_model(
            itemids=args.itemids,
            data_dir=args.data_dir,
            output_dir=output_dir,
            pore_model_names=args.pore_models,
            meltpool_model_name=args.meltpool_model,
            props=args.props,
            param_dict=param_dict,
            totalpores=args.totalpores,
            merged_data_dir=args.data_dir
        )

        print(f"Model run complete.")
        print(f"   Output saved to: {results['output_dir']}")
        print(f"   Plot saved to: {results['plot_path']}\n")

    except Exception as e:
        import traceback
        print(f"Error during model run: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    run_model_cli()
