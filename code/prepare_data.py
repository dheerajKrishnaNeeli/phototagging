#!/usr/bin/env python
# coding: utf-8

# # Summary
# This notebook is for creating new training data for the photo tagging model by aggregating Appen and Ground Truth labeled data. Each time a new training data set is created, you should create 
#a PR for your work so that other team members can review it.
#
# You should be able to run the notebook as is while filling in the appropriate parameters in certain cells.
#
# It's recommended to run this on a Sagemaker notebook.

# In[ ]:


#get_ipython().system(' pip install scikit-multilearn==0.2.0')


# In[68]:


import boto3
import json
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
from datetime import date, datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


# In[ ]:


s3 = boto3.client("s3")


# # Helper Functions

# In[71]:


def download_files(s3_paths, local_folder):
    if not os.path.isdir(local_folder):
        os.mkdir(local_folder)

    s3 = boto3.client('s3')

    for s3_path in tqdm(s3_paths):
        key = '/'.join(s3_path.split("/")[3:])
        local_path = local_folder + "/" + key.split("/")[-1]
        if os.path.exists(local_path):
            continue
        s3.download_file('cerebro-machine-learning-field-images', key, local_path)

def s3_save(file, s3_folder):
    s3 = boto3.client('s3')

    bucket = s3_folder.split('/')[0]
    key = '/'.join(s3_folder.split('/')[1:] + [file])

    s3.upload_file(
        file,
        bucket,
        key
    )

    print(f'Uploaded to: "s3://{bucket}/{key}"')

def consolidate_results(annotations, annotation_batch_last_modified):
    url, labels, last_modified = [], [], []

    for row in annotations:
        url.append(row['dataObject']['s3Uri'])
        labels.append(json.loads(row['annotations'][0]['annotationData'][
            'content'])['image-contains']['labels'])
        last_modified.append(annotation_batch_last_modified)

    df_agg = pd.DataFrame({'image_url': url, 'labels': labels, 'annotation_time': last_modified})
    df_agg.loc[:,'labels'] = (
        df_agg
        .labels
        .apply(lambda x: [i for i in x if i.lower() != 'other' and i.lower() != "i don't know" and i.lower() != "i don�t know"])
        .apply(lambda x: ",".join(x))
    )
    df_agg.loc[:, 'unique_tags'] = df_agg.labels.apply(np.unique).apply(lambda x: ",".join(x))

    # ignore all photos without any tags
    df_agg = df_agg[df_agg['labels'] != '']

    return df_agg

def replace_element(l, keys, replacement):
    result = []
    for i in l.split(","):
        if i in keys:
            result.append(replacement)
        else:
            result.append(i)
    return ','.join(result)

def replace_tags_in_df(df, tag_list, replacement):
    df = df.copy()
    df.loc[df['tag_list'].str.contains('|'.join(tag_list)), 'tag_list'] = (
        df.loc[df['tag_list'].str.contains('|'.join(tag_list)), 'tag_list']
        .apply(lambda x: replace_element(x, tag_list, replacement))
    )
    return df

def stratified_split(df, split_pct):
    X = df[['image_files']].values

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['filtered_tag_list'].str.split(','))

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, split_pct)

    train_set = df[df['image_files'].isin(X_train.reshape(-1,))]
    test_set = df[df['image_files'].isin(X_test.reshape(-1,))]

    # check for overlaps
    overlaps = list(set(train_set['image_files']).intersection(test_set['image_files']))
    train_set = train_set[~train_set['image_files'].isin(overlaps)]
    assert not set(train_set['image_files']).intersection(test_set['image_files'])

    return train_set, test_set

def process_appen_data():
    df_appen = pd.read_csv('appen_labeling_results.csv')

    df_appen.columns = ["image_url", "unique_tags"]

    # remove spaces from tags
    df_appen.loc[df_appen['unique_tags'].notnull(), 'unique_tags'] = (
        df_appen.loc[df_appen['unique_tags'].notnull(), 'unique_tags']
        .str.split(",")
        .apply(lambda x: [i.strip() for i in x])
        .apply(lambda x: ','.join(x))
    )

    # ignore photos without any tags
    df_appen = df_appen[df_appen['unique_tags'].notnull()]

    # get image basenames
    df_appen['image_files'] = df_appen.apply(lambda x: x.image_url.split("/")[-1], axis=1)

    # get image basenames
    df_appen['image_files'] = df_appen.apply(lambda x: x.image_url.split("/")[-1], axis=1)

    assert df_appen['image_url'].drop_duplicates().shape[0] == df_appen.shape[0]

    df_appen = df_appen[["image_url", "unique_tags", "image_files"]].rename(columns={'unique_tags': 'tag_list'})

    # define Appen time as 2020-08-01 before merging
    df_appen["annotation_time"] = np.datetime64('2020-08-01T00:00:00.000000000')

    # convert to UTC to avoid clash of timezone sensitive objects
    df_appen["annotation_time"] = df_appen["annotation_time"].dt.tz_localize('Europe/London')
    df_appen["annotation_time"] = df_appen["annotation_time"].dt.tz_convert("UTC")

    # edit columns
    df_appen = df_appen[["image_url", "tag_list", "annotation_time", "image_files"]]

    return df_appen

def process_gt_data(annotation_paths):
    paginator = s3.get_paginator('list_objects')

    l = []
    for path in tqdm(annotation_paths):
        pages = paginator.paginate(Bucket='cerebro-labeling-jobs', Prefix='/'.join(path.split('/')[3:]))

        for page in pages:
            try:
                contents = page['Contents']
            except KeyError:
                print('Error:', path)
                break

            for obj in contents:
                obj_dict = s3.get_object(Bucket='cerebro-labeling-jobs', Key=obj['Key'])
                annotations = json.loads(obj_dict['Body'].read().decode('utf-8'))
                last_modified_date = obj_dict['LastModified']
                df_tmp = consolidate_results(annotations, last_modified_date)
                df_tmp["annotation_source"] = path
                df_tmp['job_num'] = int(path.split('/')[4].split('-')[-1])
                l.append(df_tmp)

    df_gt = pd.concat(l)

    # get image basenames
    df_gt['image_files'] = df_gt.apply(lambda x: x.image_url.split("/")[-1], axis=1)

    df_gt = df_gt[["image_url", "unique_tags", "annotation_time", "image_files", "job_num"]].rename(columns={'unique_tags': 'tag_list'})

    return df_gt

def merge_duplicate_tags(df, tag_list):
    tags = pd.Series([j for i in df['tag_list'].to_list() for j in i.split(",")])
    tag_counts = tags.value_counts()

    tags = tag_counts[
    (tag_counts.index.str.contains("formwork"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "formwork")

    tags = tag_counts[
        (tag_counts.index.str.contains("metal")) &
        (tag_counts.index.str.contains("fram")) &
        (~tag_counts.index.str.contains("ceiling"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "metal framing")

    tags = tag_counts[
        (tag_counts.index.str.contains("ceiling")) &
        (tag_counts.index.str.contains("fram"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "ceiling framing")

    tags = tag_counts[
        (tag_counts.index.str.contains("wood")) &
        (tag_counts.index.str.contains("fram"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "wood framing")

    tags = tag_counts[
        (tag_counts.index.str.contains("mep"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "mep system")

    tags = tag_counts[
        (tag_counts.index.str.contains("electrical")) &
        (tag_counts.index.str.contains("cabinet|panel"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "electrical cabinet")

    tags = tag_counts[
        (tag_counts.index.str.contains("switches"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "switches/outlets")

    tags = tag_counts[
        (tag_counts.index.str.contains("floor")) &
        (tag_counts.index.str.contains("finish"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "floor finish")

    tags = tag_counts[
        (tag_counts.index.str.contains("tile")) &
        (~tag_counts.index.str.contains("ceiling|roof"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "tile")

    tags = tag_counts[
        (tag_counts.index.str.contains("cabinet"))
        & (~tag_counts.index.str.contains("fire|elec"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "cabinet")

    tags = tag_counts[
        (tag_counts.index.str.contains("toilet"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "toilet")

    tags = tag_counts[
        (tag_counts.index.str.contains("sink"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "sink")

    tags = tag_counts[
        (tag_counts.index.str.contains("ceiling")) &
        (tag_counts.index.str.contains("tile"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "ceiling tile")

    tags = tag_counts[
        (tag_counts.index.str.contains("tape"))
    ]
    df = replace_tags_in_df(df, tags.index.tolist(), "repair tape")

    df['filtered_tag_list'] = (
        df.tag_list
        .apply(lambda x: [i for i in x.split(",") if i in tag_list])
        .apply(lambda x: ','.join(x))
    )
    df = df[df.filtered_tag_list != '']

    return df

def get_unique_labels(training_data_df):
    # Convert str to list
    training_data_df['tags'] = training_data_df.filtered_tag_list.apply(lambda x: x.split(','))

    # Explode list
    training_data_df_xpld = training_data_df[['image_url', 'image_files', 'key', 'tags']].explode('tags')

    # Deduplicate
    training_data_df_xpld = training_data_df_xpld.drop_duplicates()

    # Collect tags into list
    deduped_training_data_df = (
        training_data_df_xpld
        .groupby(['image_url', 'image_files'])
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


# # Processing Data

# ## Aggregate Appen Data

# In[4]:
def aggregate_appen_data():

    s3.download_file(
        "cerebro-labeling-jobs",
        "photo-tagging/figure-eight-jobs/figure_eight_labeled.csv",
        "appen_labeling_results.csv",
        ExtraArgs={'VersionId': 'SKQEamdKcdQw76dECoDVIiX_7Fk9miR.'}
    )


    df_appen = process_appen_data()


    print(df_appen.shape)
    df_appen.head()
    return df_appen

# ## Aggregate GroundTruth Data

# We should only keep the latest available folder to avoid counting through the same manifest more than once since we iterate over each s3 location.

# In[1]:


all_groundtruth_annotation_paths = [
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-1/output/photo-tagging-job-1/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-2/output/photo-tagging-job-2/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-3/output/photo-tagging-job-3/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-4/output/photo-tagging-job-4/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-6/output/photo-tagging-job6-mk/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-7/output/photo-tagging-job7-mk/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-8/output/photo-tagging-job-8-v2/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-9/output/photo-tagging-job-9/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-10/output/photo-tagging-job-10-new/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-11/output/photo-tagging-job-11/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-12/output/photo-tagging-job-12-new/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-13/output/photo-tagging-job-13-new/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-14/output/photo-tagging-job-14/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-15/output/photo-tagging-job-15-chain/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-16/output/photo-tagging-job-16/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-17/output/photo-tagging-job-17-new-clone-chain-chain/annotations/consolidated-annotation/consolidation-request/'
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-18/output/photo-tagging-job-18-clone-chain-chain/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-19/output/photo-tagging-job-19-clone-chain/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-20/output/photo-tagging-job-20-clone-chain-chain/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-21/output/photo-tagging-job-21/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-23/output/photo-tagging-job-23/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-24/output/photo-tagging-job-24-chain-2/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-25/output/photo-tagging-job-25-chain-2/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-26/output/photo-tagging-job-26/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-27/output/photo-tagging-job-27/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-28/output/photo-tagging-job-28/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-29/output/photo-tagging-job-29/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-30/output/photo-tagging-job-30-copy/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-31/output/photo-tagging-job-31/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-32/output/photo-tagging-job-32/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-33/output/photo-tagging-job-33/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-34/output/photo-tagging-job-34-chain/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-35/output/photo-tagging-job-35-chain/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-36/output/photo-tagging-job-36-walls/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-38/output/photo-tagging-job-38/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-39/output/photo-tagging-job-39/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-40/output/photo-tagging-job-40/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-41/output/photo-tagging-job-41-formwork-clone2/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-42/output/photo-tagging-job-42-ductwork-clone/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-43/output/photo-tagging-job-43-drywall-clone/annotations/consolidated-annotation/consolidation-request/',
    's3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-44/output/photo-tagging-job-44-mep-clone/annotations/consolidated-annotation/consolidation-request/',
]

def split_data(df, pct_splits):
    train_pct, valid_pct, test_pct = pct_splits
    train_df, val_test_df = stratified_split(df, (100-100*train_pct)/100)
    val_df, test_df = stratified_split(val_test_df, test_pct/(valid_pct+test_pct))
    train_val_df = pd.concat([train_df.assign(is_valid=False), val_df.assign(is_valid=True)])
    overlaps = list(set(train_val_df['image_files']).intersection(test_df['image_files']))

    assert len(overlaps) == 0
    return train_val_df, test_df
# In[ ]:
def download_data_for_version(version=None, local_path=None):
    client = MlflowClient()
    run_id=None
    model_versions=client.search_model_versions("name='PhotoTagging'")
    #TODO handle 0 versions
    if (len(model_versions) == 0):
        return None
    if version is None:
        # run=mlflow.get_run(dict(model_versions[-1])['run_id'])
        run_id = dict(model_versions[-1])['run_id']
    else:
        for mv in client.search_model_versions("name='PhotoTagging'"):
            mv=dict(mv)
            if mv['version']==version:
                # run=mlflow.get_run(mv['run_id'])
                run_id = mv['run_id']
                break
        if run_id is None:
            print("Model with given version not present so we are taking latest model")
            # run=mlflow.get_run(dict(model_versions[-1])['run_id'])
            run_id = dict(model_versions[-1])['run_id']
    print("run_id for base version: " + run_id)
    return client.download_artifacts(run_id, 'model/artifacts/processed_data.csv')

def prepare_data(new_groundtruth_annotation_paths,version=None):
    # When incrementally adding new data, new-groundtruth_annotation_paths need to be specified.
    # When that is omitted, we consider it as re-generating the entire training data
    combine_old=False
    if (new_groundtruth_annotation_paths is None):
        df_old = aggregate_appen_data()
        df_gt = process_gt_data(all_groundtruth_annotation_paths)
    else:
        # TODO: Use s3 paths and try to connect this with model version so that we can download the training data
        # used for any particular model version and add to it
        # if (os.path.isfile('processed_data.csv')): # if this is the first time we are running this
        #     df_old = pd.read_csv('processed_data.csv')
        # else:
        local_artifact_path = "."
        data_file = download_data_for_version(version, local_artifact_path)
        print(data_file)
        if (data_file is not None):
            df_old = pd.read_csv(data_file).rename(columns={'filtered_tag_list': 'tag_list'})
        else:
            df_old = aggregate_appen_data()
        # df_old = aggregate_appen_data()
        df_gt = process_gt_data(new_groundtruth_annotation_paths)

        # df=pd.read_csv(run.info.artifact_uri+"/model/artifacts/processed_data.csv")
        # df_gt=[df_gt,df]


    print(df_gt.shape)
    df_gt.head()


    df = pd.concat([df_old, df_gt])
    df['key'] = df['image_url'].str.split('/').str[3:].str.join('/')
    df['image_files'] = df['key'].apply(lambda x: x.replace('/', '_'))


    print(df.shape)
    df.head()


    tag_list = [
        'bannister',
        'bathtub',
        'cabinet',
        'ceiling framing',
        'ceiling tile',
        'conduit',
        'drywall',
        'ductwork',
        'electrical cabinet',
        'floor finish',
        'formwork',
        'mep system',
        'metal framing',
        'people',
        'rebar',
        'repair tape',
        'sink',
        'stairs',
        'switches/outlets',
        'tile',
        'toilet',
        'wood framing'
    ]


    df = merge_duplicate_tags(df, tag_list)

    df = get_unique_labels(df)

    print(df.shape)
    df.head()

    df.to_csv("processed_data.csv", index=False)
    # run=mlflow.active_run()
    # df.to_csv(run.info.artifact_uri+"processed_data.csv",index=False)


    train_val_df, test_df = split_data(df, [.8, .1, .1])

    assert df.shape[0] == train_val_df.shape[0] + test_df.shape[0]



    train_val_df.to_csv("train_val_set.csv", index=False)
    test_df.to_csv("test_set.csv", index=False)


    #run=mlflow.active_run()
    #train_val_df.to_csv(run.info.artifact_uri+"train_val_df.csv",index=False)
    #test_df.to_csv(run.info.artifact_uri+"test_df.csv",index=False)


    # TODO: upload to proper buckets later
    # s3_save("train_val_set.csv", f"team-cerebro-repo-data/photo-tagging/data/processed/{notebook_name}")
    # s3_save("test_set.csv", f"team-cerebro-repo-data/photo-tagging/data/processed/{notebook_name}")


    with open('tags.json', 'w') as f:
        json.dump(sorted(tag_list), f)