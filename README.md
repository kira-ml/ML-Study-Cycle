# � ML Study Cycle – Let's Learn Machine Learning Together!

Hey there, fellow learner! 👋

Welcome to my **Machine Learning study journey**! I'm Ken, and I'm on a mission to master ML from the ground up—and I want to take you along with me. This isn't just a collection of code; it's a **living, breathing learning companion** where we explore the beautiful mathematics, intuition, and code behind machine learning.

---

## 💡 Why This Repository Exists

I believe the best way to truly understand ML is to **build it from scratch**. No magic libraries hiding the details—just pure NumPy, mathematics, and curiosity. 

I'm creating this repository because:
- 🎓 **I'm learning too!** I'm a student just like you, working through these concepts step by step
- 🤝 **Learning together is powerful** – sharing my journey helps solidify my understanding and hopefully helps yours too
- 🔥 **I'm passionate about AI** and want to make these intimidating topics accessible and exciting
- 📚 **Teaching is the best way to learn** – by explaining concepts clearly, I deepen my own mastery

Whether you're completely new to ML, coming from a different background, or just want to strengthen your foundations, **you're in the right place**. We'll struggle through the tough parts together, celebrate the "aha!" moments, and build something amazing.

> 💭 **My Philosophy**: If you can code it from scratch, you truly understand it. That's why every algorithm here is implemented without hiding behind scikit-learn or TensorFlow (at first, anyway!).

---

## 📚 What's Inside

* [Why This Repository Exists](#-why-this-repository-exists)
* [Our Learning Path](#-our-learning-path)
* [What We're Learning](#-what-were-learning)
* [How to Join This Journey](#-how-to-join-this-journey)
* [Resources I'm Using](#-resources-im-using)
* [Let's Learn Together](#-lets-learn-together)
* [License](#-license)

---

## 🗺️ Our Learning Path

I've structured this journey into digestible modules that build on each other. Think of it as our ML curriculum—created by a student, for students!

```
ML-Study-Cycle/
├── 00-math-fundamentals/      # The foundation: linear algebra, calculus, stats
├── 01-python-for-ml/          # Python skills we need for ML
├── 02-fundamentals-of-machine-learning/  # Core ML algorithms
├── 03-model-evaluation/       # How do we know if our model is good?
├── 04-feature-engineering/    # Making our data work for us
├── 05-model-optimization/     # Making our models better
├── 06-intro-to-deep-learning/ # Neural networks and beyond!
├── notebooks/                 # Interactive explorations
└── utils/                     # Helper code
```

### What Makes This Different?

📌 **From Scratch Implementation** – We code the math ourselves (no sklearn shortcuts at first!)  
📌 **Clear Explanations** – I explain things the way I wish they were explained to me  
📌 **Progressive Difficulty** – Start simple, build up gradually  
📌 **Real Code You Can Run** – Every concept has working Python code  
📌 **My Learning Notes** – See my thought process, mistakes, and discoveries

---

## 🎯 What We're Learning

Here's what we're tackling together! Each topic is something I've wrestled with, implemented, and (hopefully) understood well enough to explain.

### 📐 **Math Fundamentals** – The Language of ML

Don't let the math scare you! I break down:
- **Linear Algebra**: Vectors, matrices, transformations (the backbone of ML!)
- **Calculus**: Gradients and derivatives (how models learn!)
- **Probability & Statistics**: Understanding uncertainty and data distributions

🎓 *Why it matters*: ML is applied mathematics. Understanding the math means understanding WHY algorithms work, not just HOW to use them.

### 🐍 **Python for ML** – Our Toolbox

- NumPy mastery for array manipulation
- Vectorized operations (making code fast and elegant)
- Building algorithms from scratch (the hard but rewarding way)

### 🤖 **Core ML Algorithms** – The Classics

- **Linear & Logistic Regression** – Where it all begins
- **Gradient Descent** – The heartbeat of learning
- **Decision Trees & Random Forests** – Interpretable and powerful
- **Support Vector Machines** – Finding the best boundary
- **K-Means & Clustering** – Finding hidden patterns

🔥 *My approach*: We implement each algorithm from scratch first, THEN use libraries. This way, we know what's happening under the hood!

### 📊 **Model Evaluation** – Are We Actually Learning?

- Train/validation/test splits (the right way!)
- Cross-validation strategies
- Metrics that matter: accuracy, precision, recall, F1, MSE, MAE
- Confusion matrices and ROC curves

### 🛠️ **Feature Engineering** – The Secret Sauce

- Data preprocessing and cleaning (the unglamorous but crucial work)
- Handling missing values and outliers
- Feature scaling, normalization, encoding
- Creating new features from existing ones

### 🎛️ **Model Optimization** – Making It Better

- Hyperparameter tuning (finding the sweet spot)
- Bias-variance tradeoff (the eternal struggle)
- Regularization techniques (L1, L2, dropout)
- Learning curves and debugging strategies

### 🧠 **Deep Learning Basics** – The Exciting Frontier

- Neural network architecture (layers, neurons, connections)
- Activation functions (ReLU, sigmoid, tanh)
- Backpropagation (the magic of automatic differentiation!)
- CNNs for images, RNNs for sequences
- Attention mechanisms and Transformers (yes, we're going there!)

💪 *Challenge accepted*: I'm implementing backpropagation from scratch. It's tough, but SO rewarding when it clicks!

---

## 🚀 How to Join This Journey

Ready to learn together? Here's how to get started:

### 1️⃣ **Grab the Code**

```bash
git clone https://github.com/kira-ml/ML-Study-Cycle.git
cd ML-Study-Cycle
```

### 2️⃣ **Set Up Your Environment**

```bash
# Create a virtual environment (always a good practice!)
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 3️⃣ **Start Learning!**

I recommend starting with `00-math-fundamentals/` and working your way through, but feel free to jump around based on your interests and background!

```bash
# Start with linear algebra basics
cd 00-math-fundamentals/linear-algebra
python ex01_vector_arithmetic.py
```

### 4️⃣ **How to Use Each Module**

- 📖 **Read the code carefully** – I've added lots of comments explaining my thinking
- ✍️ **Modify and experiment** – Change parameters, break things, fix them!
- 🤔 **Don't just copy** – Type it out yourself, understand each line
- 💬 **Ask questions** – Open an issue if something's unclear!

> 💡 **Pro tip**: The best learning happens when you implement it yourself. Use my code as a reference, but try coding it from scratch first!

### 5️⃣ **Track Your Progress**

I've organized the exercises in numbered order. Work through them sequentially for the best learning experience, or cherry-pick topics you're curious about!

---

## 📖 Resources I'm Using

These are the resources that have been invaluable in MY learning journey. I highly recommend them!

### 🎥 **Video Courses & Channels**
- [**3Blue1Brown – Essence of Linear Algebra**](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) – The BEST visual intuition for linear algebra!
- [**StatQuest with Josh Starmer**](https://www.youtube.com/user/joshstarmer) – Makes complex concepts simple and fun
- [**Andrew Ng's Machine Learning Course**](https://cs229.stanford.edu/) – The classic that started it all for many of us
- [**fast.ai's Practical Deep Learning**](https://course.fast.ai/) – Top-down approach that's incredibly practical

### 📚 **Books I'm Reading**
- [**Mathematics for Machine Learning**](https://mml-book.github.io/) – Free and comprehensive math reference
- **Pattern Recognition and Machine Learning** (Bishop) – Dense but thorough
- **Deep Learning** (Goodfellow, Bengio, Courville) – The deep learning bible

### 🌐 **Other Great Resources**
- **Kaggle** – For datasets and learning from others' notebooks
- **Papers with Code** – Bridging research and implementation
- **Distill.pub** – Beautiful, interactive explanations of ML concepts

> 🤝 **My recommendation**: Don't just passively watch/read. Pause, implement, experiment. That's where real learning happens!

---

## 🤝 Let's Learn Together!

I'm on this journey too, and I'd love for you to join me! Here's how we can help each other:

### 💬 **Ask Questions**
Stuck on something? Confused by a concept? **Please open an issue!** Chances are, if you're confused, I was too (or still am!). Your questions help me improve my explanations.

### 🐛 **Found a Bug?**
My code isn't perfect (I'm learning, remember?). If you spot an error, please let me know!

### 💡 **Have an Idea?**
Want to add a new exercise? Have a better way to explain something? **Contributions are welcome!**

You can:
- 🎯 Suggest new topics or exercises
- 🔧 Improve existing code or explanations  
- 📝 Add your own notes or alternative implementations
- 🌟 Share how you used this repo in your learning journey

### 🎯 **My Goals for This Repo**

I'm constantly updating this as I learn more. Here's what's coming:
- [ ] More advanced deep learning topics (GANs, Transformers in detail)
- [ ] Reinforcement learning fundamentals
- [ ] MLOps basics (making models production-ready)
- [ ] More interactive notebooks with visualizations
- [ ] Video walkthroughs explaining key concepts

### ⭐ **Show Some Love**

If this repository helps you on your ML journey, please **star it**! It motivates me to keep learning and sharing, and it helps other learners discover this resource.

> 🌟 **Fun fact**: Every star is a reminder that we're all learning together. It's not about being perfect; it's about progress!

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE) – which means you're free to use, modify, and share it. All I ask is that you pay it forward and help other learners too!

---

## 💭 Final Thoughts

Machine learning can seem intimidating at first—trust me, I've been there! All those Greek letters, complex equations, and abstract concepts can be overwhelming. But here's what I've learned:

**Everyone starts as a beginner.** The researchers and engineers you admire? They all struggled with basic concepts at some point. The difference is they kept going.

**It's okay to not understand everything immediately.** Some concepts will click right away; others might take weeks or even months. That's completely normal.

**Implementation beats theory every time.** You can read about backpropagation all day, but you won't truly get it until you've debugged your own implementation at 2 AM.

**The journey is the reward.** ML is vast—impossibly vast. You'll never learn "everything." But every concept you master, every algorithm you implement, makes you better than yesterday.

So let's embrace the struggle, celebrate the small wins, and build something amazing together! 🚀

---

**Ready to start?** Clone the repo and let's dive into [00-math-fundamentals/linear-algebra](00-math-fundamentals/linear-algebra)!

**Have questions or want to chat about ML?** Open an [issue](https://github.com/kira-ml/ML-Study-Cycle/issues) or reach out!

Happy learning,  
**Ken** 🎓✨

*"The beautiful thing about learning is that no one can take it away from you." – B.B. King*
