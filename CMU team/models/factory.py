import pandas as pd
import numpy as np
import inspect

class PoreModelFactory:
    """
    Factory to create and manage multiple pore source model instances.

    Supports:
    - Creating single or multiple pore models by name
    - Combined sampling from multiple pore models
    - Aggregated pore density calculation
    - Aggregated number density histogram from combined samples
    """

    @staticmethod
    def create(model_name, params, *args, props = None, **kwargs):
        """
        Create a single pore model instance.

        Raises:
            ValueError: If model_name is unknown.
            TypeError: If model_name is not a string.
        """
        from models import get_model

        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        try:
            model_cls = get_model(model_name)
        except KeyError:
            raise ValueError(f"Pore model '{model_name}' is not registered.")

        if model_cls is None:
            raise ValueError(f"Pore model '{model_name}' is not registered.")

        # Inspect constructor params
        sig = inspect.signature(model_cls.__init__)
        if 'props' in sig.parameters and props is not None:
            # Pass props only if accepted and given
            return model_cls(params, *args, props=props, **kwargs)
        else:
            # Else skip props
            return model_cls(params, *args, **kwargs)

    @staticmethod
    def create_multiple(model_names, params, *args, props=None, **kwargs):
        from models import get_model

        if not hasattr(model_names, '__iter__') or isinstance(model_names, str):
            raise TypeError("model_names must be a non-string iterable of strings")

        instances = {}
        for name in model_names:
            if not isinstance(name, str):
                raise TypeError(f"Model name must be a string, got {type(name)}")
            model_cls = get_model(name)
            if model_cls is None:
                raise ValueError(f"Pore model '{name}' is not registered.")

            sig = inspect.signature(model_cls.__init__)
            if 'props' in sig.parameters and props is not None:
                instances[name] = model_cls(params, *args, props=props, **kwargs)
            else:
                instances[name] = model_cls(params, *args, **kwargs)

        return instances

    @staticmethod
    def combined_sample(models, nsamples=100000, **sample_kwargs):
        if not isinstance(models, dict):
            raise TypeError("models argument must be a dict of model_name -> model instance")

        densities = {}
        total_density = 0.0

        for name, model in models.items():
            try:
                dens = model.getPoreDensity()
                dens = dens if dens is not None and dens > 0 else 0
            except Exception:
                dens = 0
            densities[name] = dens
            total_density += dens

        if total_density == 0:
            n_models = len(models)
            if n_models == 0:
                # No models at all; return empty DataFrame with pore_source column
                return pd.DataFrame(columns=['pore_source'])
            samples_per_model = {name: max(1, nsamples // n_models) for name in models.keys()}
        else:
            samples_per_model = {
                name: max(1, int(nsamples * (dens / total_density))) for name, dens in densities.items()
            }

        sampled_dfs = []
        for name, model in models.items():
            n = samples_per_model.get(name, 0)
            if n > 0:
                try:
                    df = model.sample(n, **sample_kwargs)
                except Exception:
                    df = pd.DataFrame()
                if df is not None and not df.empty:
                    df = df.copy()
                    df['pore_source'] = name
                    sampled_dfs.append(df)

        if not sampled_dfs:
            # Return empty DataFrame with pore_source column to prevent errors downstream
            return pd.DataFrame(columns=['pore_source'])

        combined = pd.concat(sampled_dfs, ignore_index=True)
        return combined

    @staticmethod
    def combined_pore_density(models):
        if not isinstance(models, dict):
            raise TypeError("models argument must be a dict of model_name -> model instance")

        total_density = 0.0
        for model in models.values():
            try:
                dens = model.getPoreDensity()
                if dens is not None and dens > 0:
                    total_density += dens
            except Exception:
                pass
        return total_density

    @staticmethod
    def combined_number_density_histogram(models, nsamples=100000, minsize=None, maxsize=None, res=2.5, **sample_kwargs):
        combined_samples = PoreModelFactory.combined_sample(models, nsamples, **sample_kwargs)

        if combined_samples.empty or 'ESD' not in combined_samples.columns:
            # Return empty histogram and bin edges arrays
            return np.array([]), np.array([0.0])

        if minsize is None:
            minsize = combined_samples.ESD.min()
        if maxsize is None:
            maxsize = combined_samples.ESD.max()

        pore_density = PoreModelFactory.combined_pore_density(models)
        bins_count = max(1, int(np.ceil((maxsize - minsize) / res)))

        hist, bin_edges = np.histogram(combined_samples.ESD, bins=bins_count, range=(minsize, maxsize), density=True)

        # Scale histogram to actual pore density (pores per mm^3)
        hist *= pore_density

        return hist, bin_edges
