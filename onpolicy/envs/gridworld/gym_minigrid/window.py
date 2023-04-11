import sys
import numpy as np

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip3 install --user matplotlib')
    sys.exit(-1)

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None
        self.local_imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(1,2)

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        for ax in self.ax:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            _ = ax.set_xticklabels([])
            _ = ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img, local_img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax[0].imshow(img, interpolation='bilinear')
        if self.local_imshow_obj is None:
            self.local_imshow_obj = self.ax[1].imshow(local_img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.local_imshow_obj.set_data(local_img)

        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
