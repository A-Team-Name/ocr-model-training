import tkinter as tk
from tkinter import Button
from PIL import Image, ImageDraw
import numpy as np


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Image")

        # Create a 64x64 pixel canvas to draw on
        self.canvas_width = 64
        self.canvas_height = 64
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # Initialize image to store the drawing
        self.image = Image.new("1", (self.canvas_width, self.canvas_height), color=1)  # 1-bit image (black/white)
        self.draw = ImageDraw.Draw(self.image)

        # Current position to draw
        self.last_x, self.last_y = None, None
        self.pen_thickness = 3  # pen thickness

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)

        # Add 'Done' button to save the image
        self.done_button = Button(root, text="Done", command=self.done_drawing)
        self.done_button.pack()

        # Flag to ensure the window closes only after "Done"
        self.drawing_done = False

    def draw_on_canvas(self, event):
        """Draw a line on the canvas and on the image."""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=self.pen_thickness, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=self.pen_thickness)
        self.last_x, self.last_y = event.x, event.y

    def reset_position(self, event):
        """Reset the position when mouse is released."""
        self.last_x, self.last_y = None, None

    def done_drawing(self):
        """Called when 'Done' is clicked. Sets the flag and returns the image."""
        # Set drawing_done flag to True to indicate the drawing is done
        self.drawing_done = True
        self.root.quit()  # Close the window
        self.root.destroy()

    def get_image_array(self):
        """Returns the image as a numpy array once the drawing is done."""
        if self.drawing_done:
            image_array = np.array(self.image)
            return image_array
        else:
            return None  # Or handle cases where drawing isn't done yet


def draw_image() -> np.ndarray:
    """Opens the drawing window and returns the drawn image as a numpy array."""
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()  # Run the Tkinter main loop

    # After the window is closed, get the image if drawing is done
    if app.drawing_done:
        return app.get_image_array()
    else:
        return None  # Return None or handle as needed
