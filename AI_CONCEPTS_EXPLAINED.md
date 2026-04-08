# AI Concepts Explained: A Beginner's Guide to Paranormix

This document serves as a beginner-friendly guide to the Artificial Intelligence concepts used in the Paranormix project. It translates the technical code into easy-to-understand concepts suitable for an introductory AI course.

---

## 1. What is Natural Language Processing (NLP)?

At its core, Paranormix uses NLP—a field of AI that helps computers understand and process human language. Since computers only understand numbers, we have to "translate" text into math.

### A. Tokenization & Lemmatization
*   **Tokenization**: Think of this as cutting a sentence into individual words (tokens). 
*   **Lemmatization**: This is like finding the "root" of a word. For example, "ghosts," "ghosting," and "ghostly" are all reduced to "ghost." This helps the AI realize they all mean the same basic thing.

### B. N-grams (Context Matters)
A "Unigram" is a single word. A "Bigram" is a pair of words. 
*   **Example**: The word "white" could mean anything. But the Bigram **"white lady"** is a specific paranormal signal. By looking at groups of 2 words together (N-grams), the AI understands context better.

---

## 2. Turning Text into Numbers: TF-IDF

We use a technique called **TF-IDF (Term Frequency-Inverse Document Frequency)** to weight the importance of words.

*   **Term Frequency (TF)**: How often does a word appear in a story? (e.g., if "shadow" appears 5 times, it's important).
*   **Inverse Document Frequency (IDF)**: How unique is this word across all stories? Words like "the" or "and" appear everywhere, so the AI gives them a very low score. Words like "ectoplasm" are rare, so they get a high score.

**The result**: The AI focuses on the "meaningful" words that define a story.

---

## 3. The Brain: Machine Learning Classification

Paranormix uses a **Classifier** (specifically an `SGDClassifier`). Imagine a child sorting blocks into different colored buckets. The classifier does the same with stories.

### A. Training the Model
We "showed" the AI thousands of stories where we already knew the answer (e.g., "this is a poltergeist story"). The AI patterns-matched the words to the labels.

### B. Class Weights (The Human Nudge)
Sometimes, one "bucket" is harder to fill than others.
*   **Example**: In our data, "Creature" stories were rare. To make sure the AI didn't just ignore them, we gave that category a higher **Weight** (like a 2x multiplier). This tells the AI: "Pay extra attention when you see creature signals!"

---

## 4. The Interpreter: Large Language Models (LLMs)

While the classifier does the math, **Llama-3** (our LLM) does the talking.

*   **The Problem**: The classifier only gives us numbers (e.g., "0.82 poltergeist"). 
*   **The Solution**: We pass that data to Llama-3 with a "Job Description" (called a **System Prompt**). We tell it: "You are a Technical Analyst. Here are the numbers; explain them to the user like a human expert."

### A. Determinism (Temperature 0.0)
In most AI chats, you want the AI to be creative. In research, we want the AI to be **Deterministic** (factual). We set the "Temperature" to 0.0, which tells the AI: "Do not imagine things. Stick exactly to the data I gave you."

---

## 5. Explainable AI (XAI) Philosophy

Standard AI is often a "Black Box"—you put something in, and an answer pops out, but you don't know *why*.

**Paranormix is "Explainable."** 
Instead of just saying "This is a ghost," it shows you the "Evidence Signals" (which words triggered the result) and the "Confidence Bands" (how sure it is). This transparency is vital for academic research.

---

## Summary for your Course
In this project, you have combined:
1.  **Classic ML**: Using statistical patterns to categorize text.
2.  **Modern Generative AI**: Using LLMs to make that data human-readable.
3.  **Human-Centric Design**: Building a UI that visualizes the "thinking" process of the machine.

---
*Created for introductory AI studies and algorithmic transparency.*
