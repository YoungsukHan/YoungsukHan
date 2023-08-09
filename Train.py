from shoelace.dataset import LtrDataset
from shoelace.iterator import LtrIterator
from shoelace.loss.listwise import listnet
from shoelace.loss.listwise import listmle
from shoelace.loss.listwise import listpl
from chainer import training, optimizers, links, Chain
import chainer.functions as F
from chainer.training import extensions
import numpy as np
from chainer.dataset import convert
from chainer import serializers
from lambdamart import LambdaMART
import warnings
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import pickle
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

class Ranker(Chain):
    # def __init__(self, predictor, loss):
    #     super(Ranker, self).__init__(predictor=predictor)
    #     self.loss = loss
    def __init__(self, loss):
        super(Ranker, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(None, 16)
            self.l2 = links.Linear(None, 16)
            self.l3 = links.Linear(None, 8)
            self.l4 = links.Linear(None, 1)
            self.loss = loss

    # def __call__(self, x, t):
    #     x_hat = self.predictor(x)
    def __call__(self, x, t):
        # x_hat = self.predictor(x)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        x_hat = self.l4(h3)
        return self.loss(x_hat, t)

    def predict_result(self, x):
        # x_hat = self.predictor(x)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        x_hat = self.l4(h3)
        return x_hat

# def dcg_score(y_true, y_score, k=5):
#     """Discounted cumulative gain (DCG) at rank K.
#
#     Parameters
#     ----------
#     y_true : array, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array, shape = [n_samples, n_classes]
#         Predicted scores.
#     k : int
#         Rank.
#
#     Returns
#     -------
#     score : float
#     """
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#
#     gain = 2 ** y_true - 1
#
#     discounts = np.log2(np.arange(len(y_true)) + 2)
#     return np.sum(gain / discounts)
#
#
# def ndcg_score(ground_truth, predictions, k=5):
#     """Normalized discounted cumulative gain (NDCG) at rank K.
#
#     Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
#     recommendation system based on the graded relevance of the recommended
#     entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
#     ranking of the entities.
#
#     Parameters
#     ----------
#     ground_truth : array, shape = [n_samples]
#         Ground truth (true labels represended as integers).
#     predictions : array, shape = [n_samples, n_classes]
#         Predicted probabilities.
#     k : int
#         Rank.
#
#     Returns
#     -------
#     score : float
#
#     Example
#     -------
#     >>> ground_truth = [1, 0, 2]
#     >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
#     >>> score = ndcg_score(ground_truth, predictions, k=2)
#     1.0
#     >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
#     >>> score = ndcg_score(ground_truth, predictions, k=2)
#     0.6666666666
#     """
#     lb = LabelBinarizer()
#     lb.fit(range(len(predictions) + 1))
#     T = lb.transform(ground_truth)
#
#     scores = []
#
#     # Iterate over each y_true and compute the DCG score
#     for y_true, y_score in zip(T, predictions):
#         actual = dcg_score(y_true, y_score, k)
#         best = dcg_score(y_true, y_true, k)
#         score = float(actual) / float(best)
#         scores.append(score)
#
#     return np.mean(scores)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
# def dcg_score(*args):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """

    # for arg in args:
    #     print(arg)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(estimator, y_true, y_score, groups):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """

    y_true = estimator.predict(y_true)

    # array 생성
    y = np.zeros(shape=(y_score.shape[0],), dtype=np.float64)

    # 1 세팅
    start = 0
    for i in range(groups.shape[0]):
        print(i)
        if(i == 16):
            print("")
        end = groups[i]
        t = y_true[start:start+end]
        maxIndex = np.argmax(t)
        y[start+maxIndex] = 1
        start = start+end

        print("")

    # 비교

    y
    y_score

    # return actual / best


def test_linear_network():
    # To ensure repeatability of experiments
    np.random.seed(1042)

    train_flag = True
    # train_flag = False

    # '''MIP 모델링 분류'''
    # f = 0  # 원래 / 근사최적해
    # # f = 1 # newMIP / 최적해
    #
    # machineNum = 2
    # # machineNum = 5
    # # machineNum = 10
    # # machineNum = 15
    #
    # jobNum = 10
    # # jobNum = 20
    # # jobNum = 25
    # # jobNum = 50
    # # jobNum = 75
    # # jobNum = 100
    # # jobNum = 150
    #
    # # model = LambdaMART(training_data, 300, 0.001, 'sklearn')
    # # 	model.fit()
    # # 	model.save('lambdamart_model_%d' % (i))
    #
    # # modelType = "Boosting"
    # modelType = "NN"
    #
    # # LTRModel = "LambdaMART"
    # # LTRModel = "listnet"
    # LTRModel = "listmle"
    # # LTRModel = "listpl"

    # LTRModel = ["listnet", "listmle", "listpl", "LambdaMart"]
    # LTRModel = ["LambdaMart"]
    LTRModel = ["xgboost", "LightGBM"]
    # LTRModel = ["LightGBM"]

    list_ = [[0, 2, 10], [0, 2, 20], [0, 5, 25], [0, 5, 50], [0, 10, 50]
                , [0, 10, 100], [0, 15, 75], [0, 15, 150]

                , [1, 2, 10], [1, 2, 20], [1, 5, 25], [1, 5, 50], [1, 10, 50]
                # , [1, 10, 100]
                , [1, 15, 75]
                # , [1, 15, 150]
             ]

    # list_ = [[1, 15, 75]]

    # relevanceType 1 = 1or0, relevanceType 2 = 5or0, relevanceType 3 = 10or0
    relevanceType = 1

    result = []
    for e in list_:
        e.append(LTRModel)
        result.append(e)
    # result.pop()

    for f, machineNum, jobNum, LTRModels in result:

        for LTRModel in LTRModels:

            # if(LTRModel!="LambdaMart"):
            #     continue

            fileLocation = "C:\\Users\\Han\\Google 드라이브\\oneDrive Bak\\대학원\\논문\\석사졸업논문\\실험\\ScheduML_data\\"  ## File 위치
            fileLocation = fileLocation + "M" + str(machineNum) + "N" + str(jobNum)


            if f == 0:
                if relevanceType == 1:
                    fileName = '\\3.MIP_A\\3-1.ToDispatching\\MIP_A_Dispatching_training'  ## xlsx file 명
                    modelFileName = '\\3.MIP_A\\3-4.Model\\MIP_A_Dispatching_Model_' + LTRModel
                else:
                    fileName = '\\3.MIP_A\\3-1.ToDispatching\\MIP_A_Dispatching_training_relType2'  ## xlsx file 명
                    modelFileName = '\\3.MIP_A\\3-4.Model\\MIP_A_Dispatching_Model_relType'+str(relevanceType)+LTRModel
            else:
                if relevanceType == 1:
                    fileName = '\\4.MIP_O\\4-1.ToDispatching\\MIP_O_Dispatching_training'
                    modelFileName = '\\4.MIP_O\\4-4.Model\\MIP_O_Dispatching_Model_' + LTRModel
                else:
                    fileName = '\\4.MIP_O\\4-1.ToDispatching\\MIP_O_Dispatching_training_relType2'
                    modelFileName = '\\4.MIP_O\\4-4.Model\\MIP_O_Dispatching_Model_relType'+str(relevanceType)+LTRModel
            if(LTRModel.find("list") != -1):
                modelFileName = modelFileName + "l4u128"
                # Load data set
                # with open('./dataset_shoelace.txt', 'r') as f:
                fn = fileLocation + fileName + ".txt"
                with open(fn, 'r') as file:
                        dataset = LtrDataset.load_txt(file)
                # dataset = get_dataset(True)
                iterator = LtrIterator(dataset, repeat=True, shuffle=True)
                eval_iterator = LtrIterator(dataset, repeat=False, shuffle=False)

                # Create neural network with chainer and apply our loss function
                # predictor = links.Linear(None, 1)
                #
                # if LTRModel=="listnet":
                #     loss = Ranker(predictor, listnet)
                # elif LTRModel=="listmle":
                #     loss = Ranker(predictor, listmle)
                # elif LTRModel=="listpl":
                #     loss = Ranker(predictor, listpl)

                if LTRModel == "listnet":
                    loss = Ranker(listnet)
                elif LTRModel == "listmle":
                    loss = Ranker(listmle)
                elif LTRModel == "listpl":
                    loss = Ranker(listpl)

                if train_flag == True:
                    # Build optimizer, updater and trainer
                    optimizer = optimizers.Adam(alpha=0.2)
                    optimizer.setup(loss)
                    updater = training.StandardUpdater(iterator, optimizer)
                    trainer = training.Trainer(updater, (30, 'epoch'))

                    # Train neural network
                    trainer.run()
                else:
                    serializers.load_npz(fileLocation+modelFileName+'.model', loss)

                # # Evaluate loss after training
                # result = eval(loss, eval_iterator)

                if train_flag == True:
                    serializers.save_npz(fileLocation + modelFileName + '.model', loss)
                    # serializers.save_npz(fileLocation+modelFileName+'_AddScoreFunc.model', loss)

            elif LTRModel == "LambdaMart":
                # with open(fn, 'r') as file:
                fn = fileLocation + fileName + ".txt"
                training_data = get_data(fn)
                model = LambdaMART(training_data, 3, 0.001, 'sklearn')
                model.fit()
                model.save(fileLocation + modelFileName + '.model')
            elif LTRModel == "xgboost":
                #  This script demonstrate how to do ranking with xgboost.train
                x_train, y_train = load_svmlight_file(fileLocation + fileName+".train")
                group_train = []
                with open(fileLocation + fileName+".group", "r") as _file:
                    data = _file.readlines()
                    for line in data:
                        group_train.append(int(line.split("\n")[0]))

                train_dmatrix = DMatrix(x_train, y_train)
                train_dmatrix.set_group(group_train)

                params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
                          'min_child_weight': 0.1, 'max_depth': 6}

                xgb_model = xgb.train(params, train_dmatrix, num_boost_round=50)

                file_name = fileLocation + modelFileName + '.pkl'
                # save
                pickle.dump(xgb_model, open(file_name, "wb"))
            elif LTRModel == "LightGBM":
                #  This script demonstrate how to do ranking with xgboost.train
                x_train, y_train = load_svmlight_file(fileLocation + fileName + ".train")
                group_train = []
                with open(fileLocation + fileName + ".group", "r") as _file:
                    data = _file.readlines()
                    for line in data:
                        group_train.append(int(line.split("\n")[0]))

                # train_dmatrix = DMatrix(x_train, y_train)
                # train_dmatrix.set_group(group_train)

                params = {'boosting_type': 'gbdt', 'objective': 'lambdarank',
                          'colsample_bytree': 0.5, 'subsample': 1.0, 'learning_rate': 0.02,
                          'num_leaves': 100, 'max_depth': 20, 'min_child_samples': 5,
                          'n_estimators': 50, 'random_state': 42, }
                # params_grid = {'n_estimators': [10, 20, 30, 40],
                #                'num_leaves': [20, 50, 100, 200],
                #                'max_depth': [5, 10, 15, 20],
                #                'learning_rate': [0.01, 0.02, 0.03],
                #                }
                #
                # gbm = lgb.LGBMRanker(**params)
                # group_train = np.array(group_train)
                # group_info = group_train.astype(int)
                # flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
                # gkf = GroupKFold(n_splits=5)
                # cv = gkf.split(x_train, y_train, groups=flatted_group)
                # cv_group = gkf.split(x_train, groups=flatted_group)  # separate CV generator for manual splitting groups
                #
                # # generator produces `group` argument for LGBMRanker for each fold
                # def group_gen(flatted_group, cv):
                #     for train, _ in cv:
                #         yield np.unique(flatted_group[train], return_counts=True)[1]
                #
                # gen = group_gen(flatted_group, cv_group)
                # grid = RandomizedSearchCV(gbm, params_grid, n_iter=10, cv=cv, verbose=2,
                #                           scoring=ndcg_score, refit = False)
                # gbm_model = grid.fit(x_train, y_train, group=next(gen))

                gbm = lgb.LGBMRanker(**params)
                gbm_model = gbm.fit(x_train, y_train, group=group_train, verbose=True)

                file_name = fileLocation + modelFileName + '.pkl'
                # save
                pickle.dump(gbm_model, open(file_name, "wb"))
            elif LTRModel == "directRanker":
                fileLocation = "C:\\Users\\Han\\Google 드라이브\\oneDrive Bak\\대학원\\논문\\석사졸업논문\\실험\\Learning to Rank 실습\\direct-ranker-master\\"
                file_name = fileLocation + "M15N150_MIP_A_directRanker"
                model = directRanker()
                model = directRanker.load_ranker(file_name)


def get_data(file_loc):
    f = open(file_loc, 'r')
    data = []
    for line in f:
        new_arr = []
        arr = line.split(' #')[0].split()
        score = arr[0]
        q_id = arr[1].split(':')[1]
        new_arr.append(int(float(score)))
        new_arr.append(int(float(q_id)))
        arr = arr[2:]
        for el in arr:
            new_arr.append(float(el.split(':')[1]))
        data.append(new_arr)
    f.close()
    return np.array(data)

def eval(loss_function, iterator):
    """
    Evaluates the mean of given loss function over the entire batch in given
    iterator

    :param loss_function: The loss function to evaluate
    :param iterator: The iterator over the evaluation data set
    :return: The mean loss value
    """
    iterator.reset()
    # results = []
    for batch in iterator:
        input_args = convert.concat_examples(batch)
        result = loss_function.predict_result(*input_args)
        # results.append(loss_function(*input_args).data)
    # return np.mean(results)
    return result


test_linear_network()

# # Load data and set up iterator
# with open('./dataset_shoelace.txt', 'r') as f:
#     training_set = LtrDataset.load_txt(f)
# training_iterator = LtrIterator(training_set, repeat=True, shuffle=True)
#
# # Create neural network with chainer and apply loss function
# predictor = links.Linear(None, 1)
# class Ranker(Chain):
#     def __call__(self, x, t):
#         return listnet(self.predictor(x), t)
# loss = Ranker(predictor=predictor)
#
# # Build optimizer, updater and trainer
# optimizer = optimizers.Adam()
# optimizer.setup(loss)
# updater = training.StandardUpdater(training_iterator, optimizer)
# trainer = training.Trainer(updater, (40, 'epoch'))
# trainer.extend(extensions.ProgressBar())
#
# # Train neural network
# trainer.run()