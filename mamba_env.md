# Using micromamba env config vars

This is the native conda/micromamba way to persist env vars inside an environment:

```bash
# Set a variable
micromamba env config vars set MY_VAR=some_value -n myenv

# List configured variables
micromamba env config vars list -n myenv

# Unset a variable
micromamba env config vars unset MY_VAR -n myenv
```

After setting, reactivate the environment for changes to take effect:

```bash
micromamba deactivate
micromamba activate myenv
```
