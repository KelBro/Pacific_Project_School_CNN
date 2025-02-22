from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import torchvision.models as models
import os
from datetime import datetime
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp'},
    'MAX_FILE_SIZE': 16 * 1024 * 1024,
    'SECRET_KEY': os.urandom(24)
})

if not app.debug:
    file_handler = RotatingFileHandler('app.log', maxBytes=1024 * 1024)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)


@lru_cache(maxsize=3)
def load_model(model_name):
    model_config = {
        'mobilenet': (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2),
        'resnet': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        'vgg': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1)
    }
    model_class, weights = model_config[model_name]
    return model_class(weights=weights), weights


app.config['MODELS'] = {name: load_model(name)[0] for name in ['mobilenet', 'resnet', 'vgg']}
app.config['MODEL_METAS'] = {name: load_model(name)[1] for name in ['mobilenet', 'resnet', 'vgg']}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_image(image_path, model_name):
    model = app.config['MODELS'][model_name]
    meta = app.config['MODEL_METAS'][model_name]

    try:
        img = Image.open(image_path).convert('RGB')
        preprocess = meta.transforms()
        img_tensor = preprocess(img).unsqueeze(0)

        model.eval()
        with torch.inference_mode():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top5_probs, top5_ids = torch.topk(probs, 5)
        return [(meta.meta["categories"][idx.item()], prob.item()) for prob, idx in zip(top5_probs, top5_ids)]

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        raise


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            model_name = request.form.get('model', 'mobilenet')
            if model_name not in app.config['MODELS']:
                return render_template('index.html', error="Неверная модель")

            if 'file' not in request.files:
                return render_template('index.html', error="Файл не выбран")

            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error="Файл не выбран")

            if not allowed_file(file.filename):
                return render_template('index.html', error="Неподдерживаемый формат")

            file.seek(0, os.SEEK_END)
            if file.tell() > app.config['MAX_FILE_SIZE']:
                return render_template('index.html', error="Файл слишком большой (макс. 16MB)")
            file.seek(0)

            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            predictions = predict_image(save_path, model_name)

            if predictions[0][1] < 0.1:
                return render_template('index.html', error="Не удалось распознать объект")

            return render_template('index.html',
                                   predictions=predictions,
                                   model=model_name.capitalize(),
                                   image_url=save_path)

        except Image.UnidentifiedImageError:
            return render_template('index.html', error="Некорректное изображение")
        except torch.cuda.OutOfMemoryError:
            return render_template('index.html', error="Недостаточно памяти GPU")
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}")
            return render_template('index.html', error="Внутренняя ошибка сервера")

    return render_template('index.html')


@app.route('/save')
def save_results():
    try:
        data = request.args.get('data')
        filename = f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(f'static/{filename}', 'w') as f:
            f.write(data)
        return send_file(f'static/{filename}', as_attachment=True)
    except Exception as e:
        app.logger.error(f"Save error: {str(e)}")
        return "Ошибка при сохранении", 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=False)