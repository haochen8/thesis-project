from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def has_video_frame_structure(path):
        normalized = str(path).replace('\\', '/')
        return '/frames/' in normalized

    def get_video_id_from_path(path):
        normalized = str(path).replace('\\', '/')
        parts = normalized.split('/')
        # Canonical DeepfakeBench frame path: .../frames/<video>/<frame>.png
        if len(parts) >= 3 and parts[-3] == 'frames':
            return parts[-2]
        # Image-level datasets: treat each image as an independent sample.
        return parts[-1].rsplit('.', 1)[0]

    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            a = get_video_id_from_path(s)

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        if len(new_label) == 0 or len(np.unique(new_label)) < 2:
            return np.nan, np.nan
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        if np.isnan(fpr).all() or np.isnan(tpr).all():
            return np.nan, np.nan
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        diff = np.absolute(fnr - fpr)
        if np.isnan(diff).all():
            return np.nan, np.nan
        v_eer = fpr[np.nanargmin(diff)]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list and has_video_frame_structure(img_names[0]):
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
        if np.isnan(v_auc):
            v_auc = auc
    else:
        # video-level methods OR image-only datasets
        v_auc = auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
