from tkinter import Tk, Canvas, Button, TRUE, ROUND, RAISED, SUNKEN
from PIL import Image, ImageDraw

import numpy as np

class Paint(object):

  def __init__(self, predict=None, func=None):
    self.func = func
    self.predict = predict

    self.button = "up"
    self.xold = None
    self.yold = None

    root = Tk()
    self.canvas = Canvas(root, bg='black', width=400, height=400)

    self.canvas.pack()
    self.canvas.bind("<Motion>", self.mouseMove)
    self.canvas.bind("<ButtonPress-1>", self.mouseDown)
    self.canvas.bind("<ButtonRelease-1>", self.mouseUp)

    self.image = Image.new("L", (400, 400))
    self.draw = ImageDraw.Draw(self.image)
    root.mainloop()

  def mouseDown(self, event):
    self.button = "down"
    self.canvas.delete("all")

  def mouseUp(self, event):
    self.button = "up"
    self.xold = None
    self.yold = None

    x = np.array(self.image.resize((20, 20), resample=Image.BILINEAR), dtype=float).reshape(1, 400)
    x = x / 255
    if self.func:
      self.func(x)
    if self.predict:
      pred = self.predict(x)
      print(f'Predicted: {pred if pred != [10] else [0]}')

    self.image = Image.new("L", (400, 400))
    self.draw = ImageDraw.Draw(self.image)

  def mouseMove(self, event):
    if self.button == "down":
      if self.xold is not None and self.yold is not None:
        self.draw.line([(self.xold, self.yold), (event.x, event.y)], width=10, fill='white', joint='curve')
        event.widget.create_line(self.xold, self.yold, event.x, event.y, width=10, fill='white', capstyle=ROUND, smooth=TRUE)

      self.xold = event.x
      self.yold = event.y
