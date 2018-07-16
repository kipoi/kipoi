"""Plugin function
"""
import importlib
import pkgutil
import pandas as pd

PLUGINS = [
    {"plugin": "kipoi_veff",
     "url": "https://github.com/kipoi/kipoi-veff",
     "cli": True,
     "description": "Variant effect prediction"},
    {"plugin": "kipoi_interpret",
     "url": "https://github.com/kipoi/kipoi-interpret",
     "cli": True,
     "description": "Model interpretation using feature importance scores like ISM, grad*input or DeepLIFT."},
]


def is_installed(package):
    """Check if a python package (Kipoi plugin) is installed

    Args:
      package (str): package/plugin name

    Returns:
      True if package/plugin is installed
    """
    return pkgutil.find_loader(package) is not None


def load_plugin(plugin):
    """Get the plugin module
    """
    return importlib.import_module(plugin)


def list_installed(cli_support=False):
    """
    Args:
       cli_support: if True, only packages with CLI
           support will be listed
    Returns:
      list of installed plugin names
    """
    def cli_filter(plugin):
        if cli_support:
            return plugin['cli']
        else:
            return True

    return [plugin['plugin']
            for plugin in PLUGINS
            if is_installed(plugin['plugin']) and cli_filter(plugin)]


def list_plugins():
    """List available plugins

    Returns:
      pd.DataFrame with columns:
        - plugin (str) - plugin name - should correspond to the python name
        - installed (bool) - True if the plugin is installed
        - cli (bool) - True if the package supports the command-line interface
        - description (str) - short description
        - url (str) - url of the plugin
    """
    df = pd.DataFrame.from_dict(PLUGINS)
    df['installed'] = pd.Series([is_installed(plugin)
                                 for plugin in df.plugin])
    return df[['plugin', 'installed', 'cli', 'description', 'url']]


def get_model_yaml_parser(plugin):
    """
    Returns:
      Related class
    """
    return load_plugin(plugin).ModelParser


def get_dataloader_yaml_parser(plugin):
    """
    Returns:
      Related class
    """
    return load_plugin(plugin).DataloaderParser


def get_cli_fn(plugin):
    """
    Returns:
      CLI function
    """
    return load_plugin(plugin).cli_main


def get_plugin_cli_fns():
    return {plugin: get_cli_fn(plugin)
            for plugin in list_installed(cli_support=True)}


def get_plugin_help():
    header = "# Plugins"
    dfp = list_plugins()
    plugin_msg = []
    for i in range(len(dfp)):
        p = dfp.iloc[i]
        if p.installed and p.cli:
            plugin_msg.append("{}     {}".format(p.plugin, p.description))
    return "\n".join([header] + plugin_msg)
