# Weighted Deep Forest

> Apply variable-length PSO to find the optimal structure of Weighted Random Forest

## Related Projects

- [gcForest official source code](https://github.com/kingfengji/gcForest) 

## Install

```
- conda create --name py36 python=3.6
- conda install -c anaconda scikit-learn
- conda install -c conda-forge xgboost
- conda install pandas
```

## Usage
- Create file `data_helper.py`:
```
data_folder = 'path/to/data/folder'
file_list = ['abalone', 'balance',...]
```

- Run:
    - __VLPSO__:
    ```
    python demo_weighted_gcforest_vlpso.py
    ```

## License

