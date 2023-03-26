import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
#model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        
        x = F.relu(x)
        
        x = self.conv2(x)
        
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        x = F.relu(x)
        
        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output

# Define the path to the checkpoint file
checkpoint_path = 'model_checkpoint.pt'

# Load the model from the checkpoint
checkpoint = torch.load(checkpoint_path)
model = Net()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the function to preprocess the image
def preprocess_image(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the image
    img = cv2.bitwise_not(img)

    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))

    # Convert the image to a PyTorch tensor
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    return img


# Define the function to make a prediction using the model
def predict_digit(img):
    # Flip the image horizontally to correct for inversion
    img = cv2.flip(img, 1)

    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.reshape(img, (1, 1, 28, 28))
    img = img.astype(np.float32) / 255.0

    # Make a prediction using the model
    with torch.no_grad():
        output = model(torch.tensor(img))
        pred = output.argmax(dim=1)

    return pred.item()

# Define the function to capture the video feed from the webcam
def capture_video_feed():
    # Open the video capture device
    cap = cv2.VideoCapture(0)
   

    # Define the coordinates of the box
    box_x = 200
    box_y = 100
    box_width = 200
    box_height = 200

    # Define the font and font size for displaying the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7

    # Continuously capture frames from the video feed
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        # Draw the box on the frame
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 2)

        # Crop the image to the area inside the box
        cropped_frame = frame[box_y:box_y+box_height, box_x:box_x+box_width]

        # Make a prediction using the model
        output = model(torch.tensor(preprocess_image(cropped_frame)))
        probs = torch.nn.functional.softmax(output, dim=-1)[0]
        top_3_probs, top_3_indices = torch.topk(probs, 3)

        # Define the x, y, and spacing values for displaying the table
        x = box_x + box_width + 50
        y = box_y
        spacing = 30

        # Display the table header
        cv2.putText(frame, "Prediction", (x, y), font, font_size, (0, 255, 0), 2)
        cv2.putText(frame, "Probability", (x + 150, y), font, font_size, (0, 255, 0), 2)

        # Display the top 3 predictions in the table format
        for i in range(3):
            pred = top_3_indices[i].item()
            prob = top_3_probs[i].item() * 100

            # Define the y coordinate for the current row
            row_y = y + (i+1) * spacing

            # Display the prediction index and probability in the table
            cv2.putText(frame, str(pred), (x, row_y), font, font_size, (0, 255, 0), 2)
            cv2.putText(frame, f"{prob:.2f}%", (x + 150, row_y), font, font_size, (0, 255, 0), 2)

        # Display the original frame on the screen
        cv2.imshow('Webcam Feed', frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture the video feed
capture_video_feed()
