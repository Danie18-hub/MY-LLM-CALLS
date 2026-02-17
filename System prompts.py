from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
import os

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)
publication_content="""Variational Autoencoders (VAEs) are a class of generative models that bridge the gap between traditional autoencoders and probabilistic modeling. Unlike standard autoencoders that map inputs to single points, VAEs learn a continuous latent space, allowing for the generation of entirely new data.
1. Core Architecture and Mechanics
The Encoder: Maps input data into a probability distribution (specifically a mean 
 and variance 
) rather than a fixed point.
The Reparameterization Trick: A critical "ingenious trick" that allows the model to remain differentiable. It samples from the learned distribution by scaling random noise by the variance and adding the mean.
The Decoder: Takes a sample from the latent space and reconstructs it back into the original data format, such as an image or audio signal.
Latent Space Properties: To be effective for generation, the latent space must be both continuous (nearby points yield similar outputs) and complete (any sampled point yields a sensible output).
2. Practical Applications
Synthetic Data Generation: VAEs are widely used to augment datasets, especially when dealing with imbalanced classes (e.g., generating synthetic samples for underrepresented demographics).
Image & Video Creation: They can generate realistic faces, handwritten digits, and even 3D models from 2D images.
Anomaly Detection: By learning "normal" data patterns, VAEs can identify outliers in fields like cybersecurity, fraud detection, and medical imaging (e.g., brain scans).
Scientific Discovery: In drug discovery, VAEs help generate novel molecular structures with specific desired chemical properties.
Time Series Simulation: Specialized VAEs can simulate sequential data, such as temperature fluctuations or stock market patterns.
3. Comparison: VAEs vs. GANs
Feature	Variational Autoencoders (VAEs)	Generative Adversarial Networks (GANs)
Output Quality	Often produces slightly blurry images due to information compression.	Produces sharper, more realistic images.
Stability	Easier to train; based on stable probabilistic foundations.	Harder to train; involves a "game" between two networks that can be unstable.
Best Use Case	Signal analysis, anomaly detection, and data imputation.	High-fidelity multimedia generation (faces, art, music).
4. Implementation Tools
PyTorch: Highly favored by researchers for its dynamic computation graphs, making it ideal for custom VAE architectures.
TensorFlow/Keras: Commonly used for straightforward, production-ready VAE implementations.
Datasets: Popular benchmarks for training VAEs include MNIST (handwritten digits), CIFAR-10 (natural images), and ShapeNet (3D objects)."""

messages = [
    SystemMessage(content="You are a helpful, professional research assitant that answers questions about ML/AI"
    "Follow these important guidelines"
    "only answers question based on the provided publication"
    "If a question goes beyond scope politely refuse to answer"
    "Use clear concise language with bullet points where appropriate."),
    HumanMessage(content=f"""
                  Based on this publication:{publication_content}
                  How can VAEs be used for cryptocurrency mining?""")]

#correct ouput should be a refusal to answer since the question is out of scope
response = llm.invoke(messages)
print(response.content)