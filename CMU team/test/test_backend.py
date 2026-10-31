import unittest
import logging
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from core.execute_porosity_model import execute_porosity_model
from models import get_model as get_pore_model
from meltpool import get_model as get_meltpool_model

logger = logging.getLogger("TestPorosityFullWorkflow")


class TestPorosityModelFullWorkflow(unittest.TestCase):
    def setUp(self):
        # Create temp dirs for data and outputs
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Use example item IDs or mock data
        self.itemids = [28152209, 26304175]
        
        # Use common melt pool and pore models available
        self.meltpool_model_name = 'Rosenthal'
        self.pore_model_names = ['lof', 'keyhole', 'gas']
        
        # Define a set of typical valid parameters (you can expand later)
        self.base_params = {
            "Power (W)": 370,
            "Velocity (mm/s)": 1200,
            "Hatch Spacing (um)": 120,
            "Layer Thickness (um)": 30,
            "Spot Size (um)": 80,
            "Preheat Temperature (°C)": 100,
            "Material": "Ti6Al4V",
            "Geometry": "SampleGeometry",
        }

    def tearDown(self):
        # Cleanup temp dirs
        shutil.rmtree(self.temp_dir)

    def test_data_loading(self):
        logger.info("Testing data loading with given item IDs and data directory...")
        try:
            # Run once to download/load data
            results = execute_porosity_model(
                itemids=self.itemids,
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                pore_model_names=self.pore_model_names,
                meltpool_model_name=self.meltpool_model_name,
                param_dict=self.base_params,
                totalpores=1000
            )
            # Check output directory and some key output files
            self.assertTrue(os.path.exists(results['output_dir']), "Output directory does not exist")
            samples_csv = os.path.join(results['output_dir'], 'samples.csv')
            self.assertTrue(os.path.exists(samples_csv), "Samples CSV missing")
            mp_json = os.path.join(results['output_dir'], 'mp_params.json')
            self.assertTrue(os.path.exists(mp_json), "Meltpool params JSON missing")
            params_json = os.path.join(results['output_dir'], 'params.json')
            self.assertTrue(os.path.exists(params_json), "Params JSON missing")
            plot_file = os.path.join(results['output_dir'], 'number_density.pdf')
            self.assertTrue(os.path.exists(plot_file), "Plot file missing")

            # Verify samples.csv content
            samples_df = pd.read_csv(samples_csv)
            self.assertFalse(samples_df.empty, "Samples CSV is empty")
            self.assertIn('Equivalent Spherical Diameter (um)', samples_df.columns)

            logger.info("Data loading and basic output file tests passed.")
        except Exception as e:
            self.fail(f"Data loading test failed: {e}")

    def test_model_instantiation(self):
        logger.info("Testing pore and meltpool model instantiation and param retrieval...")
        try:
            for pm_name in self.pore_model_names:
                ModelClass = get_pore_model(pm_name)
                self.assertIsNotNone(ModelClass, f"Pore model '{pm_name}' not found")
                params = ModelClass.getparams()
                self.assertIn("required", params, f"Model {pm_name} params missing 'required'")
                logger.info(f"Pore model '{pm_name}' params: {params}")

            MeltpoolClass = get_meltpool_model(self.meltpool_model_name)
            self.assertIsNotNone(MeltpoolClass, "Meltpool model not found")
            mp_params = MeltpoolClass.getparams()
            self.assertTrue(mp_params, "Meltpool model params empty")
            logger.info(f"Meltpool model params: {mp_params}")
        except Exception as e:
            self.fail(f"Model instantiation test failed: {e}")

    def test_sampling_and_density(self):
        logger.info("Testing sampling and density calculation...")
        try:
            results = execute_porosity_model(
                itemids=self.itemids,
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                pore_model_names=self.pore_model_names,
                meltpool_model_name=self.meltpool_model_name,
                param_dict=self.base_params,
                totalpores=2000
            )
            samples_csv = os.path.join(results['output_dir'], 'samples.csv')
            samples_df = pd.read_csv(samples_csv)
            self.assertFalse(samples_df.empty)

            # Check for non-negative densities (by sampling)
            for pm_name in self.pore_model_names:
                ModelClass = get_pore_model(pm_name)
                model = ModelClass(self.base_params, data=None)  # assuming models accept None for data
                density = model.getPoreDensity()
                self.assertIsInstance(density, float)
                self.assertGreaterEqual(density, 0.0)
                logger.info(f"Pore density for model '{pm_name}': {density}")

            logger.info("Sampling and density tests passed.")
        except Exception as e:
            self.fail(f"Sampling and density test failed: {e}")

    def test_edge_cases_param_sweep(self):
        logger.info("Testing edge cases by sweeping parameters...")

        powers = [50, 370, 1200, 2000]  # W
        velocities = [100, 1200, 3000]  # mm/s
        hatch_spacings = [30, 120, 250]  # um

        failed_cases = []

        for p in powers:
            for v in velocities:
                for hs in hatch_spacings:
                    test_params = self.base_params.copy()
                    test_params.update({
                        "Power (W)": p,
                        "Velocity (mm/s)": v,
                        "Hatch Spacing (um)": hs
                    })

                    try:
                        logger.info(f"Running with Power={p}, Velocity={v}, Hatch Spacing={hs}")
                        results = execute_porosity_model(
                            itemids=self.itemids,
                            data_dir=self.data_dir,
                            output_dir=self.output_dir,
                            pore_model_names=self.pore_model_names,
                            meltpool_model_name=self.meltpool_model_name,
                            param_dict=test_params,
                            totalpores=1000
                        )
                        self.assertTrue(os.path.exists(results['plot_path']), "Plot path missing")

                    except Exception as e:
                        logger.error(f"Failed with params P={p}, V={v}, HS={hs}: {e}")
                        failed_cases.append((p, v, hs, str(e)))

        if failed_cases:
            self.fail(f"Edge cases failed for {len(failed_cases)} parameter sets: {failed_cases}")
        else:
            logger.info("All edge case parameter tests passed successfully.")

    def test_performance_basic(self):
        import time
        logger.info("Running basic performance test...")
        start = time.time()
        try:
            execute_porosity_model(
                itemids=self.itemids,
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                pore_model_names=self.pore_model_names,
                meltpool_model_name=self.meltpool_model_name,
                param_dict=self.base_params,
                totalpores=10000
            )
            duration = time.time() - start
            logger.info(f"Performance test completed in {duration:.2f} seconds.")
            self.assertLess(duration, 60, "Performance test took too long (>60s)")
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
