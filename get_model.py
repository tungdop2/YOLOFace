import os
import gdown
import argparse

import torch

MODEL_DICT = {
    # 'yolov5m-face': '1Sx-KEGXSxvPMS35JhzQKeRBiqC98VDDI',
    'yolov7-face': '1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo',
    'yolov7s-face': '1_ZjnNF_JKHVlq41EgEqMoGE2TtQ3SYmZ',
}

def download_model(model_name, save_dir="weights"):
    """
        Download model from Google Drive.
        Args:
            model_name: name of the model to download.
            save_dir: directory where to save the model.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    url = 'https://drive.google.com/uc?id={}'.format(MODEL_DICT[model_name])
    output = os.path.join(save_dir, model_name+'.pt')
    gdown.download(url, output, quiet=False)
    print('Model downloaded to {}'.format(output))
    
    
def prepare_model(model_name, save_dir="weights"):
    print(f"Loding model {model_name} from {save_dir}.")
    model = torch.load(f'{save_dir}/{model_name}.pt', map_location='cpu')['model']
    torch.save(model.state_dict(), f'{save_dir}/{model_name}_state_dict.pth')
    os.remove(f'{save_dir}/{model_name}.pt')
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Download model from Google Drive.')
    argparser.add_argument('model', help='Name of the model to download.')
    argparser.add_argument('-d', '--dir', help='Directory where to save the model.', default='weights')
    argparser.add_argument('-l', '--list', help='List of available models.', action='store_true')
    args = argparser.parse_args()
    
    if args.list:
        print('Available models:')
        for k,v in MODEL_DICT.items():
            print(k)
            
    else:
        assert args.model in MODEL_DICT.keys(), 'Model not available.'
        if not os.path.exists(f'{args.dir}/{args.model}.pt'):
            download_model(args.model, args.dir)
        prepare_model(args.model, args.dir)
    
    