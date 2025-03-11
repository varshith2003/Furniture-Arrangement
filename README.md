# 🏡 AI-Powered Smart Furniture Arrangement  
### **Optimize room layouts intelligently using AI & Machine Learning**  

🚀 A cutting-edge AI system that arranges furniture in a room while considering **space, obstacles, and user preferences**. This project allows users to **train their own model** and generate intelligent layouts with **minimal effort**.  


![ I have completed this project on AI  powered Furniture Arrangement as a recruitment assessment for a company.]![image](https://github.com/Varshith2003/Furniture-Arrangement/blob/main/assets/logo.png)  

---

## 📌 Project Overview
This AI-powered system assists in **efficiently arranging furniture** within a defined space. The model learns from **synthetic room layouts** and predicts **optimal furniture placements** based on:  
✔ **Room dimensions** (custom width & height)  
✔ **Furniture constraints** (types & sizes)  
✔ **Obstacles (e.g., allready  placed items, doors, windows)**  
✔ **Machine learning model trained to generate optimal furniture placements**  

### 🔧 **Key Technologies Used**  
- **Python** - Core programming language  
- **TensorFlow/Keras** - AI model training  
- **FastAPI** - Backend API for inference  
- **Streamlit** - Interactive frontend UI  
- **Matplotlib** - Visualization & rendering  
  

---

## 🚀 **Features**  
✅ **Dynamic AI-generated layouts** based on room dimensions  
✅ **User-defined furniture & obstacle placement**  
✅ **Room based furniture selection** (e.g., bed will not be placed inside kitchen)  
✅ **Custom dataset creation & model training**  
✅ **Real-time backend API for model inference**  
✅ **Interactive web-based UI for visualization**  
✅ **Obstacle-aware and space-efficient predictions**  
✅ **Supports different room types (Bedroom, Living Room, Kitchen, etc.)**  

---

## 💪 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/varshith2003/Furniture-Arrangement.git
cd furniture-arrangement
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage
### 1️⃣ Generate a Synthetic Dataset(Optional)
```bash
python create_dataset.py
```
This creates **normalized_furniture_dataset.npy** for training.

### 2️⃣ Train your own AI Model(Optional)
```bash
python Training.py
```
This will train and save the model as **`furniture_model.h5`**.

### 3️⃣ Run the backend app
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
This command launches the backend server, making the AI model accessible via API. 🚀

### 4️⃣ Run the Interactive Web App
```bash
streamlit run app.py
```
This will launch a **web-based UI** where you can:

✔ select the **room type**  
✔ Set **room dimensions**  
✔ select **furniture items**  
✔ Choose **furniture constraints**  
✔ Add **obstacles**  
✔ See AI-generated **optimized furniture placement layout**  

---

## 📸 Screenshots
### 1️⃣ Streamlit UI
![image](https://github.com/Varshith2003/Furniture-Arrangement/blob/main/assets/UI.png)


### 2️⃣ AI-Suggested Furniture Placement
![image](https://github.com/Varshith2003/Furniture-Arrangement/blob/main/assets/sample1.png)

![image](https://github.com/Varshith2003/Furniture-Arrangement/blob/main/assets/sample2.png)

![image](https://github.com/Varshith2003/Furniture-Arrangement/blob/main/assets/sample3.png)

---

## 🔬 How It Works
1️⃣ **User Inputs**: Define **room type**, **room sizes**, **furniture constraints**, and **obstacles** in the UI.  
2️⃣ **Data Processing**: The system **prepares input data** for AI prediction.  
3️⃣ **AI Model (Neural Network)**:
   - Uses a **trained model** to predict **optimal furniture placements**.
   - Takes into account **available space & obstacles**.
   - optimizes the AI outputs using a **optimizer function**  
4️⃣ **Visualization**: The **optimized layout** is displayed in the UI using a **matplotlib**.

---

# 🧠 Model Details  

## 📌 Neural Network Architecture  

### **Input Layer: 36 Neurons (Representing Room & Furniture Features)**  
The **input layer** consists of **36 neurons**, where each neuron encodes a specific feature of the room and its contents.  

### **🔗 Special Connection: Mapping Input Features to Hidden Layer 1**  
The input layer captures the **spatial characteristics** of the room using:  

| **Feature Group**  | **Number of Features** | **Description** |
|--------------------|----------------------|----------------|
| **🏠 Room Features**  | **3**  | Room dimensions (type, width, height) |
| **🚪 Obstacle Data**  | **12**  | Encodes positions & sizes of obstacles (walls, doors, windows, etc.) |
| **🛋️ Furniture Data**  | **21**  | Represents furniture type and sizes.

**📌 How Input Features Map to Hidden Layer 1**  
✔ Each neuron in **Hidden Layer 1** (**16 neurons**) is mapped in a **specialised way** with these **36 input features**.  
✔ The model **learns spatial relationships** by understanding how **obstacles and furniture interact** within the available space.  
✔ **Dense connections** help preserve dependencies between **room dimensions, obstacles, and furniture placement.**  

### **🔄 Hidden Layers**  
- **Hidden Layer 1**: **16 neurons** with **ReLU activation** for **feature extraction**.  
- **Hidden Layer 2**: **24 neurons** with **ReLU activation** for **higher-order pattern learning**.  

### **📤 Output Mapping (Furniture Placement Prediction)**  
✔ The **output layer (14 neurons)** predicts **(x, y) positions** for **7 pieces of furniture**.  
✔ Each **furniture piece** is represented by **two neurons (x, y coordinates)** for precise placement.  
✔ This **ensures the model captures spatial dependencies**, generating **realistic & optimized furniture layouts**! 🏡🚀  

---

## 📊 Training Data  

- **5000+ synthetic room layouts**  
- **Furniture & obstacle configurations**  
- **Randomized layouts for robust learning**  

---

## 🔍 Special Characteristics  

✔ **The input layer is densely connected to the first hidden layer**, allowing the model to **capture spatial dependencies & object interactions effectively**.  
✔ **ReLU activation** helps the network learn **non-linear relationships**, ensuring **better placement optimization**.  
✔ The **output layer generates continuous values** to determine **precise furniture positions**.  

This **MLP model efficiently learns to arrange furniture** while considering constraints, making it ideal for **automated space optimization**! 🚀  

---

## 🎯 **Creative Use Cases**  

✔ **🏢 Smart Office Design**  
      – Optimize office layouts for **better productivity**, ensuring efficient space utilization & ergonomic furniture placement.  

✔ **🏠 Real Estate & Virtual Staging**   
      – Generate **AI-powered furniture arrangements** for real estate listings, **helping buyers visualize spaces before purchasing**.  

✔ **🎮 Game Level Design & Simulations**  
      – Automate **room & indoor environment design** for **video games, VR simulations, and metaverse applications**.  

✔ **🏗️ Architecture & Urban Planning**  
      – Assist architects in designing **optimized home layouts** based on **spatial constraints & user preferences**.  

✔ **🛋️ Furniture Retail & E-Commerce**  
      – Enable customers to **visualize furniture placements in their homes before buying** from online stores.  

✔ **📦 Warehouse & Storage Optimization**  
      – Optimize the **placement of storage units, shelves, and furniture** in warehouses for **maximum efficiency**.  

✔ **🛠️ Smart Homes & IoT Integration**  
      – Combine with **home automation systems** to automatically **rearrange smart furniture** based on **user needs & activity patterns**.  

---

---

## 🛠️ Future Improvements
🔹  **Train a CNN based model for better spatial understanding**  
🔹  **Create 3D models of furniture and allow users to vizualize rooms in 3D**  
🔹 **Create a specialised AI Agent using Distilled and lightweight LLMs on huge real-world dataset.**

---

## 🙌 Contributing
We welcome contributions!  
🔹 Fork the repository  
🔹 Create a feature branch  
🔹 Submit a Pull Request  

---

## 📩 Contact
👤 **Sai Varshith Popuri**  
📧 varshithpopuri@gmail.com  
📚 [LinkedIn](https://www.linkedin.com/in/varshith-popuri)  

---

## ⭐ If you found this useful, give it a star! ⭐
```
🌟 GitHub Repository: https://github.com/varshith2003/furniture-arrangement ⭐