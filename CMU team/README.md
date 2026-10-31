# Porosity Modeling for Laser Powder Bed Fusion Metals

This repository provides a modular framework for modeling porosity in metal parts fabricated by Laser Powder Bed Fusion (L-PBF). It supports multiple pore formation models such as lack-of-fusion (LoF), keyhole, and gas porosity, along with melt pool thermal models. The codebase allows swapping between data-driven and physics-based pore source models. Input pore size distributions are derived from X-ray micro computed tomography (X-µCT) data.

---

## Features

- Modular pore source models: LoF, keyhole, gas pores
- Melt pool thermal models with flexible parameters
- Integration of experimental CT data for pore size distributions
- CLI and GUI interfaces for simulation and visualization
- Output includes sampled pores, parameter logs, and number density plots

---

## Installation

### Prerequisites

- Python 3.9 or later
- Recommended to use a virtual environment

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/porosity-modeling.git
cd porosity-modeling
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the porosity model with custom parameters from the command line:

```bash
python run_model.py --totalpores 5000 --params input_parameters/params.json 
```

Use `--help` to see all options. You can modify the params file to modify the process parameters.

### Graphical User Interface (GUI)
The GUI can be launched via the command line by specifying the `--gui` flag:

```bash
python run_model.py --gui 
```

This will open the interactive GUI window allowing selection of pore models, melt pool models, input parameters, browsing data directories, and running simulations interactively.

This unified approach ensures consistent backend execution whether using CLI or GUI.

### Using your own data

To you use your own data, create the data/ directory (or specify an alternative data directory). The data directory requires two files poredf.csv and parameters.csv. Poredf.csv must contain the columns Name and Equivalent Spherical Diameter (um) which give the sample name and pore sizes as separate rows. Parameters.csv must contain the columns Name,Power (W), Velocity (mm/s), Spot Size (um), Hatch Spacing (um), Layer Height (um), Geometry, Volume (mm^3), where Name corresponds to poredf, power corresponds to laser powder, velocity corresponds to scanning velocity, spot size is the laser spot size (D86), hatch spacing is the distance between adjacent stripes, layer height is the thickness of layers, geometry is a parameter to specify standard geometry, and volume is the volume of scanned material. Only the 4PB geometry is tested and working for the default dataset (although some axial data is included).

If these files are not present in the data directory, the code will download data from the two dataset publications given below which are on Figshare via the Figshare API. If your data is in a figshare database, it can also be downloaded and merged by specifying additional itemids.

## Data Sources and Citations
The pore size distributions and porosity datasets used in this work are based on the following published datasets:

* Miner, Justin; Narra, Sneha Prabha (2025). Dataset of X-Ray Micro Computed Tomography Measurements of Porosity for Nominal and Low Power Coupons Fabricated by Powder Bed Fusion-Laser Beam. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/28152209.v1

* Miner, Justin; Narra, Sneha Prabha (2024). Dataset of X-ray Micro Computed Tomography Porosity Measurements from Powder Bed Fusion - Laser Beam Ti-6Al-4V. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/26304175.v1

## Acknowledgements
This work was supported by the National Aeronautics and Space Administration (NASA) University Leadership Initiative (ULI) program under grant 80NSSC19M0123 and the Space Technology Research Institute (STRI) program under grant 80NSSC23K1342.

The authors acknowledge the use of the Materials Characterization Facility at Carnegie Mellon University supported by Grant MCF-677785.

The X-µCT machine was funded by the Army Research Laboratory Cooperative Agreement Number W911NF-20-2-0175. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the US Government. The US Government is authorized to reproduce and distribute reprints for government purposes notwithstanding any copyright notation herein.

## Testing
Extensive test scripts are included in the tests/ directory to validate pore models, melt pool models, data loading, and sampling behavior over a range of process parameters, including edge cases.

Run tests using:

```bash
pytest tests/
```

## Contributing
Contributions are welcome! Please open issues or pull requests for bugs, features, or improvements.

## License
MIT License — see LICENSE file.

## Contact
Justin Miner — jpminer@andrew.cmu.edu
Sneha Prabha Narra — snarra@andrew.cmu.edu 

Carnegie Mellon University — Department of Mechanical Engineering
