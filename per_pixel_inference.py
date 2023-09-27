from utils import mean_iou
from PIL import Image
def format_predictions(args):
    predicted_labels = []
    actual_labels = []
    all_pixel_files = os.listdir(os.path.join(args.pixel_to_class,args.dataset_name))
    for f in tqdm(all_pixel_files,total=len(all_pixel_files)):
        if '.pkl' in f:
            # load predicted 
            preds = utils.open_pkl_file(os.path.join(args.pixel_to_class,args.dataset_name),f)
            actual = np.array(Image.open(os.path.join(args.annotation_location,'val',f.replace('.pkl','.png'))))
            # load actual
            preds_reshape = np.zeros((actual.shape[0],actual.shape[1]))
            actual_shape = np.zeros((actual.shape[0],actual.shape[1]))
    
        
            for (x,y),v in preds.items():
            
                preds_reshape[x,y] = v
                actual_shape[x,y] = actual[x,y]
            
            # add both 
            predicted_labels.append(preds_reshape)
            actual_labels.append(actual_shape)
    return predicted_labels,actual_labels

def get_mean_iou(args,actual_labels,pred_labels):
    if args.ignore_zero:
        num_classes = args.num_classes -1 
        reduce_labels = True
    else:
        num_classes = args.num_classes 
        reduce_labels = False

    iou_result = mean_iou(
    results=predicted_labels,
    gt_seg_maps=actual_labels,
    num_labels=num_classes,
    ignore_index=255,
    reduce_labels=reduce_labels)
    return iou_result 

