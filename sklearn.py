import sklearn.datasets import load-iris
import panda as pd

iris = load-iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target_name[iris.target]
iris_data.head(50)