import csv
import requests
from tqdm import tqdm
from PIL import Image
import ivy
import numpy as np
import torch
import tensorflow as tf
import jax
import jax.numpy as jnp
import haiku as hk


def test_transpiling_any_code_from_one_framework_to_another():
    def jax_fn(x):
        a = jax.numpy.dot(x, x)
        b = jax.numpy.mean(x)
        return x * a + b

    jax_x = jax.numpy.array([1., 2., 3.])
    torch_x = torch.tensor([1., 2., 3.])
    torch_fn = ivy.transpile(jax_fn, source="jax", to="torch", args=(jax_x,))
    torch_fn(torch_x)


def test_running_your_code_with_any_backend():
    ivy.set_backend("jax")
    x = jax.numpy.array([1, 2, 3])
    y = jax.numpy.array([3, 2, 1])
    ivy.add(x, y)
    
    ivy.set_backend('torch')
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([3, 2, 1])
    ivy.add(x, y)
    
    ivy.unset_backend()


# using pytorch

def test_using_pytorch_any_model_from_tensorflow():
    # Get a pretrained keras model
    eff_encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    # Transpile it into a torch.nn.Module with the corresponding parameters
    noise = tf.random.normal(shape=(1, 224, 224, 3))
    torch_eff_encoder = ivy.transpile(eff_encoder, source="tensorflow", to="torch", args=(noise,))

    # Build a classifier using the transpiled encoder
    class Classifier(torch.nn.Module):
        def __init__(self, num_classes=20):
            super().__init__()
            self.encoder = torch_eff_encoder
            self.fc = torch.nn.Linear(1280, num_classes)

        def forward(self, x):
            x = self.encoder(x)
            return self.fc(x)

    # Initialize a trainable, customizable, torch.nn.Module
    classifier = Classifier()
    classifier(torch.rand((1, 244, 244, 3)))


def test_using_pytorch_any_model_from_jax():
    # Get a pretrained haiku model
    # https://github.com/unifyai/demos/blob/15c235f/scripts/deepmind_perceiver_io.py
    
    from deepmind_perceiver_io import key, perceiver_backbone

    # Transpile it into a torch.nn.Module with the corresponding parameters
    dummy_input = jax.random.uniform(key, shape=(1, 3, 224, 224))
    params = perceiver_backbone.init(rng=key, images=dummy_input)
    ivy.set_backend("jax")
    backbone = ivy.transpile(
        perceiver_backbone, source="jax", to="torch", params_v=params, kwargs={"images": dummy_input}
    )

    # Build a classifier using the transpiled backbone
    class PerceiverIOClassifier(torch.nn.Module):
        def __init__(self, num_classes=20):
            super().__init__()
            self.backbone = backbone
            self.max_pool = torch.nn.MaxPool2d((512, 1))
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(1024, num_classes)

        def forward(self, x):
            x = self.backbone(images=x)
            x = self.flatten(self.max_pool(x))
            return self.fc(x)

    # Initialize a trainable, customizable, torch.nn.Module
    classifier = PerceiverIOClassifier()
    classifier(torch.rand((1, 3, 224, 224)))


def test_using_pytorch_any_library_from_tensorflow():
    import os
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import segmentation_models as sm

    # transpile sm from tensorflow to torch
    torch_sm = ivy.transpile(sm, source="tensorflow", to="torch")

    # get some image-like arrays
    output = torch.rand((1, 3, 512, 512))
    target = torch.rand((1, 3, 512, 512))

    # and use the transpiled version of any function from the library!
    torch_sm.metrics.iou_score(output, target)


def test_using_pytorch_any_library_from_jax():
    import rax

    # transpile rax from jax to torch
    torch_rax = ivy.transpile(rax, source="jax", to="torch")

    # get some arrays
    scores = torch.tensor([2.2, 1.3, 5.4])
    labels = torch.tensor([1.0, 0.0, 0.0])

    # and use the transpiled version of any function from the library!
    torch_rax.poly1_softmax_loss(scores, labels)


def test_using_pytorch_any_library_from_numpy():
    import madmom

    # transpile madmon from numpy to torch
    torch_madmom = ivy.transpile(madmom, source="numpy", to="torch")

    # get some arrays
    freqs = torch.arange(20) * 10

    # and use the transpiled version of any function from the library!
    torch_madmom.audio.filters.hz2midi(freqs)


def test_using_pytorch_any_function_from_tensorflow():
    def loss(predictions, targets):
        return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

    # transpile any function from tf to torch
    torch_loss = ivy.transpile(loss, source="tensorflow", to="torch")

    # get some arrays
    p = torch.tensor([3.0, 2.0, 1.0])
    t = torch.tensor([0.0, 0.0, 0.0])

    # and use the transpiled version!
    torch_loss(p, t)


def test_using_pytorch_any_function_from_jax():
    def loss(predictions, targets):
        return jnp.sqrt(jnp.mean((predictions - targets) ** 2))

    # transpile any function from jax to torch
    torch_loss = ivy.transpile(loss, source="jax", to="torch")

    # get some arrays
    p = torch.tensor([3.0, 2.0, 1.0])
    t = torch.tensor([0.0, 0.0, 0.0])

    # and use the transpiled version!
    torch_loss(p, t)


def test_using_pytorch_any_function_from_numpy():
    def loss(predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

    # transpile any function from numpy to torch
    torch_loss = ivy.transpile(loss, source="numpy", to="torch")

    # get some arrays
    p = torch.tensor([3.0, 2.0, 1.0])
    t = torch.tensor([0.0, 0.0, 0.0])

    # and use the transpiled version!
    torch_loss(p, t)


# using tensorflow

def test_using_tensorflow_any_model_from_pytorch():
    import timm

    # Get a pretrained pytorch model
    mlp_encoder = timm.create_model("mixer_b16_224", pretrained=True, num_classes=0)

    # Transpile it into a keras.Model with the corresponding parameters
    noise = torch.randn(1, 3, 224, 224)
    mlp_encoder = ivy.transpile(mlp_encoder, to="tensorflow", args=(noise,))

    # Build a classifier using the transpiled encoder
    class Classifier(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = mlp_encoder
            self.output_dense = tf.keras.layers.Dense(units=1000, activation="softmax")

        def call(self, x):
            x = self.encoder(x)
            return self.output_dense(x)

    # Transform the classifier and use it as a standard keras.Model
    x = tf.random.normal(shape=(1, 3, 224, 224))
    model = Classifier()
    model(x)


def test_using_tensorflow_any_model_from_jax():
    # Get a pretrained haiku model
    # https://github.com/unifyai/demos/blob/15c235f/scripts/deepmind_perceiver_io.py
    from deepmind_perceiver_io import key, perceiver_backbone

    # Transpile it into a tf.keras.Model with the corresponding parameters
    dummy_input = jax.random.uniform(key, shape=(1, 3, 224, 224))
    params = perceiver_backbone.init(rng=key, images=dummy_input)
    backbone = ivy.transpile(
        perceiver_backbone, to="tensorflow", params_v=params, args=(dummy_input,)
    )

    # Build a classifier using the transpiled backbone
    class PerceiverIOClassifier(tf.keras.Model):
        def __init__(self, num_classes=20):
            super().__init__()
            self.backbone = backbone
            self.max_pool = tf.keras.layers.MaxPooling1D(pool_size=512)
            self.flatten = tf.keras.layers.Flatten()
            self.fc = tf.keras.layers.Dense(num_classes)

        def call(self, x):
            x = self.backbone(x)
            x = self.flatten(self.max_pool(x))
            return self.fc(x)

    # Initialize a trainable, customizable, tf.keras.Model
    x = tf.random.normal(shape=(1, 3, 224, 224))
    classifier = PerceiverIOClassifier()
    classifier(x)


def test_using_tensorflow_any_library_from_pytorch():
    import kornia

    # transpile kornia from torch to tensorflow
    tf_kornia = ivy.transpile(kornia, source="torch", to="tensorflow")

    # get an image
    url = "http://images.cocodataset.org/train2017/000000000034.jpg"
    raw_img = Image.open(requests.get(url, stream=True).raw)

    # convert it to the format expected by kornia
    img = np.array(raw_img)
    img = tf.transpose(tf.constant(img), (2, 0, 1))
    img = tf.expand_dims(img, 0) / 255

    # and use the transpiled version of any function from the library!
    tf_kornia.enhance.sharpness(img, 5)


def test_using_tensorflow_any_library_from_jax():
    import rax

    # transpile rax from jax to tensorflow
    tf_rax = ivy.transpile(rax, source="jax", to="tensorflow")

    # get some arrays
    scores = tf.constant([2.2, 1.3, 5.4])
    labels = tf.constant([1.0, 0.0, 0.0])

    # and use the transpiled version of any function from the library!
    tf_rax.poly1_softmax_loss(scores, labels)


def test_using_tensorflow_any_library_from_numpy():
    import madmom

    # transpile madmom from numpy to tensorflow
    tf_madmom = ivy.transpile(madmom, source="numpy", to="tensorflow")

    # get some arrays
    freqs = tf.range(20) * 10

    # and use the transpiled version of any function from the library!
    tf_madmom.audio.filters.hz2midi(freqs)


def test_using_tensorflow_any_function_from_pytorch():
    def loss(predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    # transpile any function from torch to tensorflow
    tf_loss = ivy.transpile(loss, source="torch", to="tensorflow")

    # get some arrays
    p = tf.constant([3.0, 2.0, 1.0])
    t = tf.constant([0.0, 0.0, 0.0])

    # and use the transpiled version!
    tf_loss(p, t)


def test_using_tensorflow_any_function_from_jax():
    def loss(predictions, targets):
        return jnp.sqrt(jnp.mean((predictions - targets) ** 2))

    # transpile any function from jax to tensorflow
    tf_loss = ivy.transpile(loss, source="jax", to="tensorflow")

    # get some arrays
    p = tf.constant([3.0, 2.0, 1.0])
    t = tf.constant([0.0, 0.0, 0.0])

    # and use the transpiled version!
    tf_loss(p, t)


def test_using_tensorflow_any_function_from_numpy():
    def loss(predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

    # transpile any function from numpy to tensorflow
    tf_loss = ivy.transpile(loss, source="numpy", to="tensorflow")

    # get some arrays
    p = tf.constant([3.0, 2.0, 1.0])
    t = tf.constant([0.0, 0.0, 0.0])

    # and use the transpiled version!
    tf_loss(p, t)


# using jax


def test_using_jax_any_model_from_pytorch():
    import timm

    # Get a pretrained pytorch model
    mlp_encoder = timm.create_model("mixer_b16_224", pretrained=True, num_classes=0)

    # Transpile it into a hk.Module with the corresponding parameters
    noise = torch.randn(1, 3, 224, 224)
    mlp_encoder = ivy.transpile(mlp_encoder, source="torch", to="jax", args=(noise,))

    # Build a classifier using the transpiled encoder
    class Classifier(hk.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.encoder = mlp_encoder()
            self.fc = hk.Linear(output_size=num_classes, with_bias=True)

        def __call__(self, x):
            x = self.encoder(x)
            x = self.fc(x)
            return x

    def _forward_classifier(x):
        module = Classifier()
        return module(x)

    # Transform the classifier and use it as a standard hk.Module
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key=rng_key, shape=(1, 3, 224, 224), dtype=jax.numpy.float32)
    forward_classifier = hk.transform(_forward_classifier)
    params = forward_classifier.init(rng=rng_key, x=x)

    forward_classifier.apply(params, None, x)


def test_using_jax_any_model_from_tensorflow():
    jax.config.update("jax_enable_x64", True)
    
    # Get a pretrained keras model
    eff_encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    # Transpile it into a hk.Module with the corresponding parameters
    noise = tf.random.normal(shape=(1, 224, 224, 3))
    hk_eff_encoder = ivy.transpile(eff_encoder, source="tensorflow", to="jax", args=(noise,))

    # Build a classifier using the transpiled encoder
    class Classifier(hk.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.encoder = hk_eff_encoder()
            self.fc = hk.Linear(output_size=num_classes, with_bias=True)

        def __call__(self, x):
            x = self.encoder(x)
            x = self.fc(x)
            return x

    def _forward_classifier(x):
        module = Classifier()
        return module(x)

    # Transform the classifier and use it as a standard hk.Module
    rng_key = jax.random.PRNGKey(42)
    dummy_x = jax.random.uniform(key=rng_key, shape=(1, 224, 224, 3))
    forward_classifier = hk.transform(_forward_classifier)
    params = forward_classifier.init(rng=rng_key, x=dummy_x)

    forward_classifier.apply(params, None, dummy_x)


def test_using_jax_any_library_from_pytorch():
    import kornia
    jax.config.update("jax_enable_x64", True)

    # transpile kornia from torch to jax
    jax_kornia = ivy.transpile(kornia, source="torch", to="jax")

    # get an image
    url = "http://images.cocodataset.org/train2017/000000000034.jpg"
    raw_img = Image.open(requests.get(url, stream=True).raw)

    # convert it to the format expected by kornia
    img = jnp.transpose(jnp.array(raw_img), (2, 0, 1))
    img = jnp.expand_dims(img, 0) / 255

    # and use the transpiled version of any function from the library!
    jax_kornia.enhance.sharpness(img, 5)


def test_using_jax_any_library_from_tensorflow():
    import os
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import segmentation_models as sm

    # transpile sm from tensorflow to jax
    jax_sm = ivy.transpile(sm, source="tensorflow", to="jax")

    # get some image-like arrays
    key = jax.random.PRNGKey(23)
    key1, key2 = jax.random.split(key)
    output = jax.random.uniform(key1, (1, 3, 512, 512))
    target = jax.random.uniform(key2, (1, 3, 512, 512))

    # and use the transpiled version of any function from the library!
    jax_sm.metrics.iou_score(output, target)


def test_using_jax_any_library_from_numpy():
    import madmom

    # transpile madmon from numpy to jax
    jax_madmom = ivy.transpile(madmom, source="numpy", to="jax")

    # get some arrays
    freqs = jnp.arange(20) * 10

    # and use the transpiled version of any function from the library!
    jax_madmom.audio.filters.hz2midi(freqs)


def test_using_jax_any_function_from_pytorch():
    def loss(predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    # transpile any function from torch to jax
    jax_loss = ivy.transpile(loss, source="torch", to="jax")

    # get some arrays
    p = jnp.array([3.0, 2.0, 1.0])
    t = jnp.array([0.0, 0.0, 0.0])

    # and use the transpiled version!
    out = jax_loss(p, t)


def test_using_jax_any_function_from_tensorflow():
    def loss(predictions, targets):
        return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

    # transpile any function from tf to jax
    jax_loss = ivy.transpile(loss, source="tensorflow", to="jax")

    # get some arrays
    p = jnp.array([3.0, 2.0, 1.0])
    t = jnp.array([0.0, 0.0, 0.0])

    # and use the transpiled version!
    jax_loss(p, t)


def test_using_jax_any_function_from_numpy():
    jax.config.update('jax_enable_x64', True)

    def loss(predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

    # transpile any function from numpy to jax
    jax_loss = ivy.transpile(loss, source="numpy", to="jax")

    # get some arrays
    p = jnp.array([3.0, 2.0, 1.0])
    t = jnp.array([0.0, 0.0, 0.0])

    # and use the transpiled version!
    jax_loss(p, t)


def test_using_numpy_any_library_from_pytorch():
    import kornia

    # transpile kornia from torch to np
    np_kornia = ivy.transpile(kornia, source="torch", to="numpy")

    # get an image
    url = "http://images.cocodataset.org/train2017/000000000034.jpg"
    raw_img = Image.open(requests.get(url, stream=True).raw)

    # convert it to the format expected by kornia
    img = np.transpose(np.array(raw_img), (2, 0, 1))
    img = np.expand_dims(img, 0) / 255

    # and use the transpiled version of any function from the library!
    np_kornia.enhance.sharpness(img, 5)


def test_using_numpy_any_library_from_tensorflow():
    import os
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import segmentation_models as sm

    # transpile sm from tensorflow to numpy
    np_sm = ivy.transpile(sm, source="tensorflow", to="numpy")

    # get some image-like arrays
    output = np.random.rand(1, 3, 512, 512).astype(dtype=np.float32)
    target = np.random.rand(1, 3, 512, 512).astype(dtype=np.float32)

    # and use the transpiled version of any function from the library!
    np_sm.metrics.iou_score(output, target)


def test_using_numpy_any_library_from_jax():
    import rax

    # transpile rax from jax to numpy
    np_rax = ivy.transpile(rax, source="jax", to="numpy")

    # get some arrays
    scores = np.array([2.2, 1.3, 5.4])
    labels = np.array([1.0, 0.0, 0.0])

    # and use the transpiled version of any function from the library!
    np_rax.poly1_softmax_loss(scores, labels)


def test_using_numpy_any_function_from_pytorch():
    def loss(predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    # transpile any function from torch to numpy
    np_loss = ivy.transpile(loss, source="torch", to="numpy")

    # get some arrays
    p = np.array([3.0, 2.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])

    # and use the transpiled version!
    np_loss(p, t)


def test_using_numpy_any_function_from_tensorflow():
    def loss(predictions, targets):
        return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

    # transpile any function from tf to numpy
    np_loss = ivy.transpile(loss, source="tensorflow", to="numpy")

    # get some arrays
    p = np.array([3.0, 2.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])

    # and use the transpiled version!
    np_loss(p, t)


def test_using_numpy_any_function_from_jax():
    def loss(predictions, targets):
        return jnp.sqrt(jnp.mean((predictions - targets) ** 2))

    # transpile any function from jax to numpy
    np_loss = ivy.transpile(loss, source="jax", to="numpy")

    # get some arrays
    p = np.array([3.0, 2.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])

    # and use the transpiled version!
    np_loss(p, t)


def test_using_ivy():
    # A simple image classification model
    class IvyNet(ivy.Module):
        def __init__(
            self,
            h_w=(32, 32),
            input_channels=3,
            output_channels=512,
            num_classes=2,
            data_format="NCHW",
            device="cpu",
        ):
            self.h_w = h_w
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.num_classes = num_classes
            self.data_format = data_format
            super().__init__(device=device)

        def _build(self, *args, **kwargs):
            self.extractor = ivy.Sequential(
                ivy.Conv2D(self.input_channels, 6, [5, 5], 1, "SAME", data_format=self.data_format),
                ivy.GELU(),
                ivy.Conv2D(6, 16, [5, 5], 1, "SAME", data_format=self.data_format),
                ivy.GELU(),
                ivy.Conv2D(16, self.output_channels, [5, 5], 1, "SAME", data_format=self.data_format),
                ivy.GELU(),
            )

            self.classifier = ivy.Sequential(
                # Since the padding is "SAME", this would be image_height x image_width x output_channels
                ivy.Linear(self.h_w[0] * self.h_w[1] * self.output_channels, 512),
                ivy.GELU(),
                ivy.Linear(512, self.num_classes),
            )

        def _forward(self, x):
            x = self.extractor(x)
            # flatten all dims except batch dim
            x = ivy.flatten(x, start_dim=1, end_dim=-1)
            logits = self.classifier(x)
            probs = ivy.softmax(logits)
            return logits, probs
    
    
    ivy.set_backend("torch")
    model = IvyNet()
    x = torch.randn(1, 3, 32, 32)
    model(x)
    ivy.previous_backend()
    
    ivy.set_backend("tensorflow")
    model = IvyNet()
    x = tf.random.uniform(shape=(1, 3, 32, 32))
    model(x)
    ivy.previous_backend()
    
    ivy.set_backend("jax")
    key = jax.random.PRNGKey(0)
    model = IvyNet()
    x = jax.random.uniform(key, shape=(1, 3, 32, 32))
    model(x)
    ivy.previous_backend()
    
    ivy.set_backend("numpy")
    model = IvyNet()
    x = np.random.uniform(size=(1, 3, 32, 32))
    model(x)
    ivy.previous_backend()
    
    ivy.set_backend("torch")
    model = IvyNet()

    # helper function for loading the dataset in batches
    def generate_batches(images, classes, dataset_size, batch_size=32):
        if batch_size > dataset_size:
            raise ivy.utils.exceptions.IvyError("Use a smaller batch size")
        for idx in range(0, dataset_size, batch_size):
            yield images[idx : min(idx + batch_size, dataset_size)], classes[
                idx : min(idx + batch_size, dataset_size)
            ]


    # helper function to get the number of current predictions
    def num_correct(preds, labels):
        return (preds.argmax() == labels).sum().to_numpy().item()


    # define a loss function
    def loss_fn(params):
        v, model, x, y = params
        _, probs = model(x, v=v)
        return ivy.cross_entropy(y, probs), probs


    # train the model on gpu if it's available
    device = "gpu:0" if ivy.gpu_is_available() else "cpu"

    # training hyperparams
    optimizer = ivy.Adam(1e-4)
    batch_size = 4
    num_epochs = 20
    num_classes = 10

    model = IvyNet(
        h_w=(28, 28),
        input_channels=1,
        output_channels=120,
        num_classes=num_classes,
        device=device,
    )

    images = ivy.random_uniform(shape=(16, 1, 28, 28))
    classes = ivy.randint(0, num_classes - 1, shape=(16,))


    # training loop
    def train(images, classes, epochs, model, device, num_classes=10, batch_size=32):
        # training metrics
        epoch_loss = 0.0
        metrics = []
        dataset_size = len(images)

        for epoch in range(epochs):
            train_correct = 0
            train_loop = tqdm(
                generate_batches(images, classes, len(images), batch_size=batch_size),
                total=dataset_size // batch_size,
                position=0,
                leave=True,
            )

            for xbatch, ybatch in train_loop:
                xbatch, ybatch = xbatch.to_device(device), ybatch.to_device(device)

                # Since the cross entropy function expects the target classes to be in one-hot encoded format
                ybatch_encoded = ivy.one_hot(ybatch, num_classes)

                # update model params
                loss_probs, grads = ivy.execute_with_gradients(
                    loss_fn,
                    (model.v, model, xbatch, ybatch_encoded),
                )

                model.v = optimizer.step(model.v, grads["0"])

                batch_loss = ivy.to_numpy(loss_probs[0]).mean().item()  # batch mean loss
                epoch_loss += batch_loss * xbatch.shape[0]
                train_correct += num_correct(loss_probs[1], ybatch)

                train_loop.set_description(f"Epoch [{epoch + 1:2d}/{epochs}]")
                train_loop.set_postfix(
                    running_loss=batch_loss,
                    accuracy_percentage=(train_correct / dataset_size) * 100,
                )

            epoch_loss = epoch_loss / dataset_size
            training_accuracy = train_correct / dataset_size

            metrics.append([epoch, epoch_loss, training_accuracy])

            train_loop.write(
                f"\nAverage training loss: {epoch_loss:.6f}, Train Correct: {train_correct}",
                end="\n",
            )


    # assuming the dataset(images and classes) are already prepared in a folder
    train(
        images,
        classes,
        num_epochs,
        model,
        device,
        num_classes=num_classes,
        batch_size=batch_size,
    )


def test_diving_deeper():
    ivy.set_backend("jax")

    # Simple JAX function to transpile
    def test_fn(x):
        return jax.numpy.sum(x)

    x1 = ivy.array([1., 2.])
    
    # Arguments are available -> transpilation happens eagerly
    eager_graph = ivy.transpile(test_fn, source="jax", to="torch", args=(x1,))

    # eager_graph is now torch code and runs efficiently
    eager_graph(x1)
    
    # Arguments are not available -> transpilation happens lazily
    lazy_graph = ivy.transpile(test_fn, source="jax", to="torch")

    # The transpiled graph is initialized, transpilation will happen here
    lazy_graph(x1)

    # lazy_graph is now torch code and runs efficiently
    lazy_graph(x1)


def test_ivy_as_a_framework_functional():
    def mse_loss(y, target):
        return ivy.mean((y - target)**2)

    jax_mse   = mse_loss(jnp.ones((5,)), jnp.ones((5,)))
    tf_mse    = mse_loss(tf.ones((5,)), tf.ones((5,)))
    np_mse    = mse_loss(np.ones((5,)), np.ones((5,)))
    torch_mse = mse_loss(torch.ones((5,)), torch.ones((5,)))


def test_ivy_as_a_framework_stateful():
    class Regressor(ivy.Module):
        def __init__(self, input_dim, output_dim):
            self.input_dim = input_dim
            self.output_dim = output_dim
            super().__init__()

        def _build(self, *args, **kwargs):
            self.linear0 = ivy.Linear(self.input_dim, 128)
            self.linear1 = ivy.Linear(128, self.output_dim)

        def _forward(self, x):
            x = self.linear0(x)
            x = ivy.functional.relu(x)
            x = self.linear1(x)
            return x
    
    ivy.set_backend('torch')  # set backend to PyTorch (or any other backend!)

    model = Regressor(input_dim=1, output_dim=1)
    optimizer = ivy.Adam(0.3)

    n_training_examples = 2000
    noise = ivy.random.random_normal(shape=(n_training_examples, 1), mean=0, std=0.1)
    x = ivy.linspace(-6, 3, n_training_examples).reshape((n_training_examples, 1))
    y = 0.2 * x ** 2 + 0.5 * x + 0.1 + noise


    def loss_fn(v, x, target):
        pred = model(x, v=v)
        return ivy.mean((pred - target) ** 2)

    for epoch in range(40):
        # forward pass
        model(x)

        # compute loss and gradients
        loss, grads = ivy.execute_with_gradients(lambda params: loss_fn(*params), (model.v, x, y))

        # update parameters
        model.v = optimizer.step(model.v, grads)

        # print current loss
        print(f'Epoch: {epoch + 1:2d} --- Loss: {ivy.to_numpy(loss).item():.5f}')

    print('Finished training!')