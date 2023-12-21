# agg_wflow (Aggregation Workflow)

## Overview
agg_wflow provides a comprehensive toolkit designed to simplify data visualization tasks. Consisting of Python and Bash utilities, the toolkit is tuned to work seamlessly with configuration files. This allows users to effortlessly generate plots and conduct statistical analysis directly from the command line, enhancing productivity and enabling automation.

## Requirements
- Python 3.x
- Operating System: Unix-like
- METcalcpy
- METplotpy
- METdataio

## Usage
Export the necessary environment variables
```bash

export METPLOTPY="/path/to/METplotpy"
export METCALCPY="/path/to/METcalcpy"

export PYTHONPATH="/path/to/METdataio/METdbLoad:/path/to/METdataio/METdbLoad/ush:${PTYHONPATH}"

```

### Using shell wrapper
Make sure to set the environment variables in the environment.yaml file under agg_flow/wrapper.
```bash
cd agg_flow/wrapper
python ./run_stat.py
```

If needed, adjust the configuration files under agg_flow/config.



Performance Diagram
PODY and 1 - FAR (Success Ratio)

ROC
PODY and POFD
