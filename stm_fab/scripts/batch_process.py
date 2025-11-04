# stm_fab/scripts/batch_process.py
import argparse
from stm_fab.analysis.batch_rampdown import process_folder, generate_comparison_figures, export_summary_excel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with LabVIEW .txt files")
    ap.add_argument("--excel", help="Path to save analysis_summary.xlsx")
    args = ap.parse_args()

    results = process_folder(args.input)
    if args.excel:
        export_summary_excel(results, args.excel)
    # Optionally: show figs, or save to png if needed

if __name__ == "__main__":
    main()
