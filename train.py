from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch, yolov8'n' = nano

# Use the model
results = model.train(data="config_omicron.yaml", epochs=150)  # train the model


# For resuming a past training: -----------------------------------------------
# model = YOLO("runs\detect\train6\weights\last.pt") #indicate file location
# model.train(resume=True, epochs=25) #amount of extra epochs

