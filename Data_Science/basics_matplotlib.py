import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
from mpl_toolkits import mplot3d

def basic_plot(x: np.ndarray , y_list: np.ndarray):
    """
    This function plots a list of y-values against a common x-value.

    Parameters:
    x (numpy.ndarray): A 1D numpy array representing the x-values. This array should have the same length as each y-value array in y_list.
    y_list (numpy.ndarray): A 2D numpy array where each row represents a set of y-values corresponding to the x-values. The number of columns in this array should match the length of the x-value array.

    Returns:
    None. The function displays the plot using matplotlib.pyplot.show().

    Example:
    >>> x = np.arange(1, 101, 1)
    >>> y1 = x ** 2 + 5
    >>> y2 = np.tan(x) * 20
    >>> basic_plot(x, np.array([y1, y2]))
    """
    # Iterate over each set of y-values in y_list
    for i, y in enumerate(y_list):
        # Plot the y-values against the x-values
        plt.plot(x, y, label = f"Function Y{i}")

    # Set the label for the x-axis
    plt.xlabel("Function X")
    # Set the label for the y-axis
    plt.ylabel("Function Y")
    # Set the title of the plot
    plt.title(f'Plot of Function Y against X')
    # Add a legend to the plot
    plt.legend(loc = "upper left")
    # Display the plot
    plt.show()

def subplot_plot(x: np.ndarray, y_lists: np.ndarray):
    """
    This function plots a list of y-values against a common x-value using subplots.

    Parameters:
    x (numpy.ndarray): A 1D numpy array representing the x-values. This array should have the same length as each y-value array in y_list.
    y_lists (numpy.ndarray): A 2D numpy array where each row represents a set of y-values corresponding to the x-values. The number of columns in this array should match the length of the x-value array.

    Returns:
    None. The function displays the plots using matplotlib.pyplot.show().
    Example:
    >>> x = np.arange(1, 101, 1)
    >>> y1 = x ** 2 + 5
    >>> y2 = np.tan(x) * 20
    >>> subplot_plot(x, np.array([y1, y2]))
    """

    for i, y_list in enumerate(y_lists):
        # Calculate the grid dimensions based on the number of y-value arrays.
        grid_dimensions = [
            math.ceil(math.sqrt(len(y_list))), 
            math.ceil(len(y_list) / math.ceil(math.sqrt(len(y_list))))
            ]

        # Create a subplot grid with the calculated dimensions for a figure
        fig, axs = plt.subplots(grid_dimensions[0], grid_dimensions[1])

        # set Title for the figure
        fig.suptitle(f'Plot of Functions against X')

        # Plot each set of y-values against the x-values in the corresponding subplot.
        for i in range(grid_dimensions[0]):
            for j in range(grid_dimensions[1]):
                if i*grid_dimensions[1] + j < len(y_list):
                    axs[i, j].plot(x, y_list[i*grid_dimensions[1] + j])
                    axs[i, j].set_title(f'Function {i*grid_dimensions[1] + j+1}')

    # Adjust the spacing and margins of the subplots for better visualization.
    plt.tight_layout() 
    # show all figures
    plt.show()

def bargraph_plot():
    """
    This function plots a bar graph to display the marks of multiple students in different subjects.

    Parameters:
    None.

    Returns:
    None. The function displays the plot using matplotlib.pyplot.show().

    Example:
    >>> bargraph_plot()
    """

    # Create a 2D numpy array to represent the marks of students in different subjects.
    marks = np.random.randint(0, 100, (4, 4))

    # List of student names
    names = ["Mark", "Anna", "John", "Bob"]

    # List of subject names
    subjects = ["Python", "Maths", "English", "Hindi"]

    # Plot a bar chart for all subject marks for each subject and each person.
    # The marks for each person are displayed side by side.
    for i in range(len(names)):
        # Calculate the x-coordinates for the bars of each student.
        x_index = np.arange(len(subjects)) + i*0.2

        # Plot the bars for each student.
        plt.bar(x_index, marks[i], width=0.2)

    # Set the labels for the x-axis.
    plt.xticks(np.arange(len(subjects)) + 0.3, subjects)

    # Set the label for the y-axis.
    plt.ylabel("Marks")

    # Set the title of the plot.
    plt.title("Marks in Subject")

    # Set the limits for the y-axis.
    plt.ylim(0, 120)

    # Add a legend to the plot.
    plt.legend(names, loc="upper right")

    # Display the plot.
    plt.show()

def piechart_plot():
    """
    This function plots a pie chart to display the inventory of items.

    Parameters:
    None.

    Returns:
    None. The function displays the plot using matplotlib.pyplot.show().

    Example:
    >>> piechart_plot()
    """

    # List of items in the inventory
    inventory = ["Eggs", "Milk", "Juice", "Chips"]

    # List of corresponding amounts of each item
    amount = [20000, 10000, 5000, 1234]

    # List of fractions of the radius to offset each wedge from the center of the pie
    explode = [0.1, 0, 0, 0]

    # Plot the pie chart
    plt.pie(amount, 
            labels=inventory,  # Labels for each wedge
            shadow=True,  # Whether to draw a shadow beneath the pie
            labeldistance=1.1,  # Distance of the labels from the pie chart
            autopct="%.2f%%",  # Format for the percentage labels
            explode=explode,  # Fraction of the radius to offset each wedge
            counterclock=False,  # Whether to draw wedges counterclockwise
            startangle=45,  # Angle at which to start drawing the first wedge
            )

    # Set the title of the plot
    plt.title("Inventory")

    # Display the plot
    plt.show()

def histogram_plot():
    """
    This function plots a histogram to visualize the distribution of a given dataset.

    Parameters:
    None.

    Returns:
    None. The function displays the plot using matplotlib.pyplot.show().

    Example:
    >>> histogram_plot()
    """

    # Define the parameters for the distribution
    alpha = 121
    beta = 12

    # Generate a random dataset from a normal distribution with the given parameters
    x = alpha + beta + np.random.randn(10000)

    # Plot the histogram
    plt.hist(
        x, 
        bins=100,  # Number of bins to use for the histogram
        # range=(50, 100),  # Optional: Limit the range of values to be displayed on the x-axis
        density=True,  # Optional: Normalize the histogram to display probabilities instead of counts
        histtype='bar',  # Optional: Type of histogram to be drawn (e.g., 'bar', 'barstacked', 'step', 'stepfilled')
        color='red',  # Optional: Color of the histogram bars
        )

    # Add a grid to the plot for better visualization
    plt.grid(True)

    # Display the plot
    plt.show()

def scatter_plot():
    """
    This function plots a scatter plot to visualize the relationship between two sets of data.

    Parameters:
    None.

    Returns:
    None. The function displays the plot using matplotlib.pyplot.show().

    Example:
    >>> scatter_plot()
    """

    # Generate random x and y coordinates
    x = np.random.rand(50)
    y = np.random.rand(50)

    # Plot the scatter plot
    plt.scatter(x, y,
                s= 50,  # Optional: Size of the scatter plot markers
                c= 'red',  # Optional: Color of the scatter plot markers
                edgecolors="black",  # Optional: Color of the marker edges
                )

    # Display the plot
    plt.show()

def threeD_plot():
    """
    This function creates a 3D plot using matplotlib.

    Parameters:
    None.

    Returns:
    None. The function displays the plot using matplotlib.pyplot.show().

    Example:
    >>> threeD_plot()
    """

    # Create a 3D axes object
    axs = plt.axes(projection="3d")

    # Generate a list of numbers from 0 to 20 with 100 steps
    z = np.linspace(0, 20, 100)

    # Calculate the sine and cosine values for each z value
    x  = np.sin(z)
    y  = np.cos(z)

    # Plot the 3D line using the calculated x, y, and z values
    axs.plot3D(x, y, z, 'red')

    # Set the title of the plot
    axs.set_title("3D Plot")

    # Set the label for the x-axis
    axs.set_xlabel("Sine function for z")

    # Set the label for the y-axis
    axs.set_ylabel("Cos function for z")

    # Set the label for the z-axis
    axs.set_zlabel("list of numbers")

    # Display the plot
    plt.show()

def main():

    # Example usage
    x = np.arange(1, 101, 1)
    y1 = x ** 2 + 5
    y2 = np.tan(x) * 20
    y3 = np.cos(x)
    y4 = np.log(x)
    y5 = np.exp(x)
    y6 = np.sin(x)
    y7 = np.sqrt(x)
    y8 = np.power(x, 3)
    y9 = np.cbrt(x)

    y_lists  = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9])

    # set the style for the pyplot.
    print(style.available) 
    style.use("fast")

    # basic_plot(x, np.split(y_lists, 3)[0])
    # subplot_plot(x, np.split(y_lists, 3))
    # bargraph_plot()
    # piechart_plot()
    # histogram_plot()
    # scatter_plot()
    threeD_plot()

if __name__ == '__main__':
    main()
