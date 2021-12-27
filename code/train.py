#import csvfile_create
import boto3
import ast
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastai.vision.all import *
from sklearn.metrics import classification_report , f1_score
from PIL import ImageFile
from PIL import JpegImagePlugin
import mlflow
import sys
import os.path
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from prepare_data import *
from _io import UnsupportedOperation
class MLFlowMetricLogger(Callback):
    def __init__(self):
        Recorder()
        self.best=[]
    def after_epoch(self):
        # if hasattr(self, "gather_preds"): return
        # self.path.parent.mkdir(parents=True, exist_ok=True)
        # self.old_logger,self.learn.logger = self.logger,self._write_line
        for met in self.learn.metrics:
            mlflow.log_metric(met.name, met.value, self.learn.epoch)

    # def _write_line(self, log):
    #     mlflow.log_metric("train_loss", float(log[1]))
    #     mlflow.log_metric("vaild_loss", float(log[2]))
    #     mlflow.log_metric("precision_score", float(log[3]))
    #     mlflow.log_metric("recall_score", float(log[4]))
    #     if self.best==[] or float(self.best[2])<float(log[2]):
    #         self.best=log
    # def after_fit(self):
    #     mlflow.log_metric("train_loss", float(self.best[1]))
    #     mlflow.log_metric("vaild_loss", float(self.best[2]))
    #     mlflow.log_metric("precision_score", float(self.best[3]))
    #     mlflow.log_metric("recall_score", float(self.best[4]))

class EMWAWeightAveraging(Callback):
    run_valid = False

    def __init__(self, a=0.5):
        self.a = a

    def before_step(self):
        self.weights = self.model.parameters()
        # print(self.weights)

    def after_batch(self):
        for old_weights, new_weights in zip(self.weights, self.model.parameters()):
            new_weights.detach().copy_((1-self.a)*old_weights.detach() + self.a*new_weights.detach())
            self.weights = self.model.parameters()

class PhotoTagging_Model(mlflow.pyfunc.PythonModel):
    def __init__(self):
        mlflow.pyfunc.PythonModel.__init__(self)
    def predict(context, model_input):
        raise UnsupportedOperation("Prediction is unsupported for this model")

def download_images(s3_paths, local_folder):


    if not os.path.isdir(local_folder):
        os.mkdir(local_folder)

    errors = []
    for s3_path in tqdm(s3_paths):
        # Use S3 key when forming local path to make sure names are unique
        local_path = local_folder + "/" + s3_path.replace('/', '_')

        # skip files that are already downloaded
        if os.path.exists(local_path):
            continue

        try:
            s3.download_file('cerebro-machine-learning-field-images', s3_path, local_path)
        except:
            errors.append(s3_path)
    return errors
def s3_save(file, s3_folder):
    bucket = s3_folder.split('/')[0]
    key = '/'.join(s3_folder.split('/')[1:] + [file])

    s3.upload_file(
    file,
    bucket,
    key
    )

    print(f'Uploaded to: "s3://{bucket}/{key}"')


def classification_report_to_df(report_string):
    report_data = []
    lines = report_string.split('\n')
    for line in lines[2:-1]:
        row = {}
        row_data = [i for i in line.split('  ') if i != '']

        if not row_data:
            continue

        row['class'] = row_data[0].strip()
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)

    return pd.DataFrame.from_dict(report_data)

def color_code_delta(val):
    if val < 0:
        color = 'red'
    elif val == 0:
        color = 'gray'
    else:
        color = 'green'
    return 'color: %s' % color

def bold(val):
    return 'font-weight: bold'

def get_comparison_df(old, new):
    combined = new.merge(old, on='class', suffixes=['_new', '_old'])

    combined['precision_delta'] = combined['precision_new'] - combined['precision_old']
    combined['recall_delta'] = combined['recall_new'] - combined['recall_old']
    combined['f1_score_delta'] = combined['f1_score_new'] - combined['f1_score_old']

    combined['precision_delta_rel_pct'] = round(
        (combined['precision_new'] - combined['precision_old']) / combined['precision_old'], 2
    )*100
    combined['recall_delta_rel_pct'] = round(
        (combined['recall_new'] - combined['recall_old']) / combined['recall_old'], 2
    )*100
    combined['f1_score_delta_rel_pct'] = round(
    (combined['f1_score_new'] - combined['f1_score_old']) / combined['f1_score_old'], 2
    )*100

    return (
    combined
    .rename(columns={'precision_new': 'precision', 'recall_new': 'recall', 'f1_score_new': 'f1_score'})
    [['class',
      'precision', 'recall', 'f1_score',
      'precision_delta', 'recall_delta', 'f1_score_delta',
      'precision_delta_rel_pct', 'recall_delta_rel_pct', 'f1_score_delta_rel_pct'
     ]]
    )
def add_mep_when_ductwork(df, tag_column):

    df[tag_column] = df[tag_column].map(lambda x: x+',mep system' if 'ductwork' in x else x)
    return df

def format_comparison_df(comparison_df, cols):
    return (
    comparison_df
    .style
    .set_precision(2)
    .applymap(color_code_delta, subset=pd.IndexSlice[:, cols])
    .applymap(bold, subset=pd.IndexSlice[len(comparison_df)-4:, :])
    )

def plot_thresholds(y_true, y_pred, score_func):
    scores = []
    for thresh in np.arange(0, 1, .1):
        scores.append(score_func(y_true, np.where(y_pred > thresh, 1, 0)))

    plt.plot(np.arange(0, 1, .1), scores)

    return scores

def get_unique_labels(training_data_df, is_test=False):

    # Convert str to list

    training_data_df['tags'] = training_data_df.filtered_tag_list.apply(lambda x: x.split(','))


    col_list = ['image_url', 'image_files','key']
    # Explode list
    if not is_test:
        col_list += ['is_valid']
    training_data_df_xpld = training_data_df[col_list + ['tags']].explode('tags')

    # Deduplicate
    training_data_df_xpld = training_data_df_xpld.drop_duplicates()

    # Collect tags into list
    deduped_training_data_df = (
    training_data_df_xpld
    .groupby(col_list)
    ['tags']
    .apply(list)
    .to_frame()
    .reset_index()
    )

    # Convert list to expected string representation
    deduped_training_data_df['filtered_tag_list'] = (
    deduped_training_data_df
    .tags
    .apply(lambda x: str(x).replace("'", '').replace(", ", ',')[1:-1])
    )

    deduped_training_data_df = deduped_training_data_df.drop(columns=['tags'])

    return deduped_training_data_df

def train_model(learn, epochs, lr, notebook_name, experiment_name):
    metrics_path = f'metrics/{experiment_name}_train-metrics.csv'
    # metrics_file=open(metrics_path,"r+")
    # metrics_file.truncate(0)
    # metrics_file.close()
    # with mlflow.start_run() as run:
    learn.fine_tune(
    epochs=epochs,
    freeze_epochs=epochs,
    base_lr=lr,
    cbs=[
         ShowGraphCallback(),
        CSVLogger(fname=metrics_path, append=True),
        SaveModelCallback(
            monitor='valid_loss',
            fname=f'{experiment_name}'
        ),
        EarlyStoppingCallback(
            monitor='valid_loss',
            min_delta=0.0005,
            patience=5,
        ),
         MLFlowMetricLogger(),
        ]


    )
    mlflow.log_param("epochs",epochs)
    mlflow.log_param("base_lr",lr)
    mlflow.log_param("freeze_epochs",epochs)
    # mlflow.log_artifact(f'models/{experiment_name}.pth')
    # mlflow.log_artifact(metrics_path)

    return learn

def setup_model(train_val_set):
    learn = (
        cnn_learner(
            train_val_set,
            resnet50,
            metrics=[
                PrecisionMulti(average='micro'),
                RecallMulti(average='micro'),
            ],
            cbs=EMWAWeightAveraging(a=0.5),
        )
    )

    learn.to_fp16();
    learn.model = torch.nn.DataParallel(learn.model)
    return learn
def __train(learn, notebook_name, experiment_name, epochs=2):
# Train model

    lr_min, lr_steep = learn.lr_find();
    model = train_model(learn, epochs, lr_steep, notebook_name, experiment_name)
    return model

# In[ ]:


# Tune thresholds
def tune_thresholds(learn, train_val_set, experiment_name):
    learn.load(experiment_name);

    y_preds_val, y_true_val = learn.get_preds(dl=train_val_set.valid, reorder=False)
    scores = plot_thresholds(y_true_val, y_preds_val, partial(f1_score, average='micro'))
    thresh = np.arange(0, 1, .1)[np.argmax(scores)]
    return thresh

# In[ ]:


# Evaluate on hold-out set
def evaluate(learn, experiment_name, vocab, dl, thresh):
#y_preds_test, y_true_test = learn.get_preds(dl=test_set[0], reorder=False)
    y_preds_test, y_true_test = learn.get_preds(dl=dl, reorder=False)
    report = classification_report(y_true_test, np.where(y_preds_test > thresh, 1, 0), target_names=vocab)
    results = classification_report_to_df(report)

    return results
    #results.to_csv(f'metrics/{experiment_name}_class-val-metrics.csv', index=False)

def check_readiness(results):
    tag_metrics = results.iloc[:-4].copy()
    lower_thresh_tags = tag_metrics["class"].isin(['ductwork', 'formwork', 'drywall'])
    threshold_75 = ((tag_metrics["precision"] >= 0.75) & (tag_metrics["recall"] >= 0.75))
    threshold_70 = ((tag_metrics["precision"] >= 0.7) & (tag_metrics["recall"] >= 0.7))

    ready_tags = pd.concat(
        [
            tag_metrics[(lower_thresh_tags) & threshold_70],
            tag_metrics[(~lower_thresh_tags) & threshold_75]
        ]
    )

    print(f"=> {len(ready_tags)} tags are ready <=")
    ready_tags


# # Comparison

# In[ ]:

def compare_with_baseline(experiment_name):
# Search through previous hold out metrics

    [
        i['Key']
        for i in
        s3.list_objects_v2(
            Bucket='team-cerebro-repo-data',
            Prefix='photo-tagging/models'
        )['Contents']
        if "metrics.csv" in i['Key']
    ]


# In[ ]:


    baseline_metrics_s3_key = 'photo-tagging/models/2021-05-20-vs-21-tags-model-without-mep-tag/metrics/photo-tagging_resnet50_20210520_21tags_mixup_class-val-metrics.csv'# 'photo-tagging/models/...'
    best_results_csv = f'metrics/{experiment_name}_class-val-metrics.csv'# 'metrics/...'


# ---

# In[ ]:


    s3.download_file('team-cerebro-repo-data', baseline_metrics_s3_key, 'baseline.csv')


# In[ ]:


    baseline = pd.read_csv('baseline.csv')
    best_results = pd.read_csv(best_results_csv)

    comparison_df = get_comparison_df(best_results, baseline)
    format_comparison_df(comparison_df.iloc[:, :-3], ['precision_delta', 'recall_delta', 'f1_score_delta'])

    return comparison_df

def load_data(bs):
    #run=mlflow.active_run()
    #train_val_df=pd.read_csv(run.info.artifact_uri+"train_val_df.csv")
    #test_df=pd.read_csv(run.info.artifact_uri+"test_df.csv")
    train_val_df=pd.read_csv("train_val_set.csv")
    train_val_df['key'] = train_val_df['image_url'].str.split('/').str[3:].str.join('/')
    test_df=pd.read_csv("test_set.csv")
    test_df['key'] = test_df['image_url'].str.split('/').str[3:].str.join('/')
    train_val_df = add_mep_when_ductwork(train_val_df, 'filtered_tag_list')

    train_val_df = get_unique_labels(train_val_df)

    assert train_val_df.shape == train_val_df.drop_duplicates(subset=['image_url']).shape
    test_df = add_mep_when_ductwork(test_df, 'filtered_tag_list')
    test_df = get_unique_labels(test_df,True)
    assert test_df.shape == test_df.drop_duplicates(subset=['image_url']).shape
    train_val_df=train_val_df.sample(n=96, random_state=1)
    test_df=test_df.sample(n=32,random_state=1)

    image_folder = "data/images"
    for col in train_val_df.columns:
        print(col)
    #print("length of train is ",len(train_val_df['key'].values))
    #print("length of test is ",len(test_df['key'].values))
    all_photo_paths = np.concatenate([train_val_df['key'].values, test_df['key'].values])
    errors = download_images(all_photo_paths, image_folder)
    print(len(errors))

    train_val_df = train_val_df[~train_val_df.key.isin(errors)]
    test_df = test_df[~test_df.key.isin(errors)]
    # bs = 32

    vocab = train_val_df['filtered_tag_list'].str.split(',').explode().value_counts().index.tolist()

    train_val_set = (
    ImageDataLoaders.from_df(
        train_val_df,
        path='.',
        folder=image_folder,
        fn_col='image_files',
        valid_col='is_valid',
        label_delim=',',
        label_col=train_val_df.columns.get_loc("filtered_tag_list"),
        y_block=MultiCategoryBlock(vocab=vocab),
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224, max_zoom=1), Normalize.from_stats(*imagenet_stats)],
        bs=bs,
        num_workers = 0
    )
    )

    test_set = (
    ImageDataLoaders.from_df(
        test_df,
        path='.',
        folder=image_folder,
        fn_col='image_files',
        valid_pct=0,
        label_delim=',',
        label_col=test_df.columns.get_loc("filtered_tag_list"),
        y_block=MultiCategoryBlock(vocab=vocab),
        item_tfms=Resize(224),
        batch_tfms=Normalize.from_stats(*imagenet_stats), # no augmentations in test
        bs=bs,
        num_workers = 0
    )
    )
    return train_val_set, test_set
def run_steps(new_data, base_version=None):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    JpegImagePlugin._getmp = lambda x: None
    mlflow.set_tracking_uri("http://MLFlo-MLFLO-14IPP88PBEXIM-9c5d2ec42272adee.elb.us-east-1.amazonaws.com")
    #mlflow.set_tracking_uri("http://deplo-mlflo-1358863vhzbaq-469d8c84ca3e685b.elb.us-east-2.amazonaws.com/")
    #mlflow.set_tracking_uri("s3://mlflow-artifacts-189893516120/0")
    print(mlflow.get_tracking_uri())
    s3 = boto3.client('s3')
    experiment_name = 'phototagging-train-mlflow'
    mlflow.set_experiment(experiment_name)
    import matplotlib.pyplot as plt
    from matplotlib import rcParams


    # print([
    # i['Key']
    # for i
    # in s3.list_objects_v2(
    #     Bucket='team-cerebro-repo-data',
    #     Prefix='photo-tagging/data/processed'
    # )['Contents']
    # if ("processed_data_" in i['Key'])
    # ])
    #input_array=sys.argv[1].split(",")

    with mlflow.start_run() as run:
        mlflow.log_param("new_data", new_data)
        mlflow.log_param("base_version", base_version)
        prepare_data(new_data, base_version)
        bs = 32
        mlflow.log_param("batch_size", bs)
        train_val_set, test_set = load_data(bs)
        learner = setup_model(train_val_set)

        __train(learner, 'phototagging-train-mlflow', experiment_name, 5)
        threshold = tune_thresholds(learner, train_val_set, experiment_name)
        results = evaluate(learner, experiment_name, train_val_set.vocab, test_set[0], threshold)
        results.to_csv(f'metrics/{experiment_name}_class-val-metrics.csv', index=False)
        mlflow.log_artifact(f'metrics/{experiment_name}_class-val-metrics.csv')
        rcParams.update({'figure.autolayout': True})
        ax = results.plot.barh(y=['precision', 'recall', 'f1_score'], x='class')
        ax.set_title('Classification Report')
        # ax.set_yticklabels(labels = train_val_set.vocab)
        plt.savefig("classification_report.png")
        # mlflow.log_artifact("processed_data.csv")
        # mlflow.log_artifact("train_val_set.csv")
        # mlflow.log_artifact("test_set.csv")

        # plt.show()
        mlflow.log_artifact("classification_report.png")
        check_readiness(results)
        comparison_df = compare_with_baseline(experiment_name)
        comparison_df.to_csv(f'metrics/{experiment_name}_comparison.csv', index=False)
        mlflow.log_artifact(f'metrics/{experiment_name}_comparison.csv')
        mlflow.pyfunc.log_model("model", python_model = PhotoTagging_Model(),
                                pip_requirements="requirements.txt",
                                artifacts={'model': f'models/{experiment_name}.pth', 'data': 'processed_data.csv', 'train_set': 'train_val_set.csv', 'test_set':'test_set.csv'})

if __name__=="__main__":
    # new_data = ['s3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-1/output/photo-tagging-job-1/annotations/consolidated-annotation/consolidation-request/',
    # 's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-2/output/photo-tagging-job-2/annotations/consolidated-annotation/consolidation-request/']
    # new_data=['s3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-3/output/photo-tagging-job-3/annotations/consolidated-annotation/consolidation-request/',
    # 's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-4/output/photo-tagging-job-4/annotations/consolidated-annotation/consolidation-request/']
    '''new_data = ['s3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-6/output/photo-tagging-job6-mk/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-7/output/photo-tagging-job7-mk/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-8/output/photo-tagging-job-8-v2/annotations/consolidated-annotation/consolidation-request/'
        ]'''
    temp=os.environ.get('INPUT_DATA',"hey")
    print(temp,type(temp))
    #temp1=ast.literal_eval(temp)
    temp1=temp.split(",")
    temp1[0]=temp1[0][1:]
    temp1[-1]=temp1[-1][:len(temp1[-1])-1]
    print(temp1,type(temp1))
    new_data=temp1
    base_version=os.environ.get('VERSION',"None")
    if base_version=="None":
        base_version=None
    run_steps(new_data,base_version)