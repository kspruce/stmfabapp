# hook-pySPM.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from pySPM package
datas = collect_data_files('pySPM')

# Ensure all submodules are included
hiddenimports = collect_submodules('pySPM')
