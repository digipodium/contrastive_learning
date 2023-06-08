import gradio as gr
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("keras-io/supervised-contrastive-learning-cifar10")

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def infer(test_image):
    image = tf.constant(test_image)
    image = tf.reshape(image, [-1, 32, 32, 3])
    pred = model.predict(image)
    pred_list = pred[0, :]
    return {labels[i]: float(pred_list[i]) for i in range(10)}


image = gr.inputs.Image(shape=(32, 32))
label = gr.outputs.Label(num_top_classes=3)


article = """<center>

</center>
"""


description = """Classification with a model trained via Contrastive Learning """


Iface = gr.Interface(
    fn=infer,
    inputs=image,
    outputs=label,
    examples=[["examples/cat.jpg"], ["examples/ship.jpeg"]],
    title="Contrastive Learning Classification",
    article=article,
    description=description,
).launch()