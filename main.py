import pickle
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import telebot
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

API_TOKEN = "API_TOKEN"  # Secret API TOKEN of the bot
BOT_NAME = "BOT_NAME"  # Secret BOT NAME
x = []  # X axis list
y = []  # Y axis list
z = []  # Z axis list
string_x = [] # X axis list
string_y = [] # Y axis
CHAT_BOT = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")  # Start the ai of the bot
trainer = ChatterBotCorpusTrainer(CHAT_BOT)  # Initializing the trainer for the training of the bot
trainer.train("chatterbot.corpus.english")  # Train the bot with english language with the trainer
n = 1  # Number of plots default would be one
helper_txt = "".join(open("help.txt", "r").readlines())
new_line = "\n"
labels = []
cmaps_arr = np.array(
    ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
     'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
     'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
     'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd',
     'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
     'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
     'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr',
     'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone',
     'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
     'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
     'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
     'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot',
     'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno',
     'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
     'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r',
     'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r',
     'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
     'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
)


def different_color_generator(size: int) -> list:
    """
    This function used to generate different colors based on the
    rgb color scheme

    :param size: The size of the colors list
    :return: The colors list which contains rgb in integer values
    """

    colors = []  # list to store the random rgba colors
    for i in range(size):
        # RGBA values will be between 0 and 1
        # append it to the list colors
        colors.append((random.uniform(random.uniform(0, 1), random.uniform(0, 1)),
                       random.uniform(random.uniform(0, 1), random.uniform(0, 1)),
                       random.uniform(random.uniform(0, 1), random.uniform(0, 1))
                       ))
    return colors  # Return colors list


def clear_list() -> None:
    """
    This function is used to clear all the values from x and y list
    This function will be executed whenever when a instruction is terminated
    or if the instructions has been ended

    :return: None
    """
    x.clear()
    y.clear()
    labels.clear()


def normal_graph(figure_size: tuple = (6, 4), dpi: int = 100, lw: int = 5, ls: str = "-", title: str = "Chart",
                 legend: bool = True, grid: bool = True, grid_color: str = "blue", grid_dashes=None,
                 grid_facecolor: str = "w", xlabel: str = "x", ylabel: str = "y") -> (bool, str):
    """
    Used to generate a normal 2 Dimensional cartesian graph. This function can create as many as plots in a single
    graph

    :param figure_size: A Tuple of size 2  default = (6,4)
    :param dpi: Dots per inch default = 100
    :param lw: Linewidth of the graph default lw = 5
    :param ls: Linestyle of the graph default ls = '-'
    :param title: Title of the graph default title = "Chart"
    :param legend: Legend whether to be printed or not default True
    :param grid: Grid of the graph to be visible or not default True
    :param grid_color: Color of the grid default "blue"
    :param grid_dashes: Dash patterns of the grid default will be [5,2,1,2]
    :param grid_facecolor: The background color of the grid default "white"
    :param xlabel: X axis label used to mark outside the graph default "x"
    :param ylabel: Y axis label used to mark outside the graph default "y"
    :return: (bool, str) returns True and an empty string if there was no error while creating graph else False and the
             error code and string
    """
    if grid_dashes is None:
        grid_dashes = [5, 2, 1, 2]
    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi
    )
    axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    color = different_color_generator(len(x))
    for i, j, k, l in zip(x, y, color, labels):
        try:
            axes.plot(
                i,  # X axis
                j,  # Y axis
                label=l,  # Label used in legends
                c=k,
                lw=lw,  # Linewidth of the graph
                ls=ls  # Linestyle of the graph
            )
        except Exception as e:
            return False, f"There was some error while creating graph\nError :\n{new_line.join(e.args)}"
    axes.set_xlabel = xlabel
    axes.set_ylabel = ylabel
    axes.grid(  # Creating Grid
        grid,  # Setting the visibility to True
        color=grid_color,  # Color of the grid
        dashes=grid_dashes  # Dashes pattern range
    )
    axes.set_facecolor(grid_facecolor)  # Set the facecolor i.e background

    if legend:
        axes.legend()  # Show legend
    axes.set_title(title)  # Set the title of the graph
    figure.savefig("Graph.jpeg", dpi=dpi)  # Save the image

    return True, "Graph.jpeg"


def generate_pie_chart(figure_size: tuple = (6, 4), dpi: int = 100, decimal_pointer: str = "%1.0f%%",
                       shadow: bool = True, start_angle: int = 140, textprops=None, title: str = "Pie Chart") -> (
        bool, str):
    if textprops is None:
        textprops = dict(color='w')
    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi
    )
    sample = np.arange(1, x[0].size + 1)
    axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    # Generate the color for the samples length
    colors = different_color_generator(x[0].size)
    # Explode for the length of the samples
    explode = [0.0] * x[0].size
    # Change the values so that it can explode outside the pie chart
    explode[random.randrange(0, x[0].size)] = 0.2
    try:
        wedges, txt, autotxt = plt.pie(
            x[0],  # The values of random generated number
            explode=explode,  # exploding list
            labels=sample,
            colors=colors,  # Colors list
            autopct=decimal_pointer,  # The floating point
            shadow=shadow,  # Shadow of the pie chart
            startangle=start_angle,  # Starting angle of the pie chart
            textprops=textprops  # text props of type dictionary
        )
        axes.legend(
            wedges,  # Wedges
            sample,  # The sample
            bbox_to_anchor=(0.8, 0, 0.5, 1)  # Where should be the box should be placed
        )
    except Exception as e:
        return False, f"There was some error while creating pie chart\nError :\n{new_line.join(e.args)}"
    axes.set_title(title)  # Set the title of the graph
    figure.savefig("PieGraph.jpeg", dpi=dpi)  # Save the image
    return True, "PieGraph.jpeg"


def generate_histograms(figure_size: tuple = (6, 4), dpi: int = 100, bins: str = "auto", density: bool = True,
                        stacked: bool = True, title: str = "Histogram", histtype: str = "step"):
    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi
    )
    axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    color = different_color_generator(1)[0]
    try:
        axes.hist(
            x[0],
            bins=bins,
            density=density,  # Density to be True
            stacked=stacked,  # Stack the bars True
            color=color,
            histtype=histtype  # Type of the histogram
        )
    except Exception as e:
        return False, f"There was some error while creating histogram\nError :\n{new_line.join(e.args)}"
    axes.set_title(title)
    figure.savefig("HisGraph.jpeg", dpi=dpi)
    return True, "HisGraph.jpeg"


def figure_for_3d(figure_size, dpi):
    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi
    )
    axes = figure.add_axes(
        [0, 0.1, 1, 0.9],
        projection='3d'
    )
    axes.view_init(45, 55)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    return figure, axes


def generate_3dwireframe(figure_size: tuple = (6, 4), dpi: int = 100) -> (bool, str):
    figure, axes = figure_for_3d(figure_size, dpi)

    color = different_color_generator(1)[0]
    try:
        axes.plot_wireframe(
            x[-1],
            y[-1],
            z[-1],
            color=color
        )
    except Exception as e:
        return False, f"There was some error while creating 3d wireframe plot\nError :\n{new_line.join(e.args)}"

    figure.savefig("wireframe.jpeg", dpi=dpi)
    return True, "wireframe.jpeg"


def generate_3dcontour(figure_size: tuple = (6, 4), dpi: int = 100) -> (bool, str):
    figure, axes = figure_for_3d(figure_size, dpi)
    try:
        axes.contour3D(
            x[-1],
            y[-1],
            z[-1],
            500,
            cmap=cmaps_arr[random.randint(0, cmaps_arr.size - 1)]
        )
    except Exception as e:
        return False, f"There was some error while creating 3d contour plot\nError :\n{new_line.join(e.args)}"

    figure.savefig("contour.jpeg", dpi=dpi)
    return True, "contour.jpeg"


def generate_3dsurface(figure_size: tuple = (6, 4), dpi: int = 100) -> (bool, str):
    figure, axes = figure_for_3d(figure_size, dpi)
    try:
        axes.plot_surface(  # Creating plot surface
            x,  # X data
            y,  # Y data
            z,  # Z data
            rstride=1,  # Downscaling stride in each direction.
            cstride=1,  # Downscaling stride in each direction.
            cmap=cmaps_arr[random.randint(0, len(cmaps_arr) - 1)],  # Color maps
            edgecolor="none"  # Edgecolor
        )
    except Exception as e:
        return False, f"There was some error while creating 3d surface plot\nError :\n{new_line.join(e.args)}"
    figure.savefig("surface.jpeg", dpi=dpi)
    return True, "surface.jpeg"


def generate_scatter(figure_size: tuple = (6, 4), dpi: int = 100) -> (bool, str):
    figure = plt.figure(
        figsize=figure_size,
        dpi=dpi
    )
    axes = figure.add_axes(
        [0.1, 0.1, 0.8, 0.8]
    )
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    color = different_color_generator(x[-1].size)
    try:
        axes.scatter(
            x[-1],
            y[-1],
            c=color
        )
        figure.savefig("scatter.jpeg", dpi=dpi)
    except Exception as e:
        return False, f"There was some error while creating scatter plot\nError :\n{new_line.join(e.args)}"

    return True, "scatter.jpeg"


def get_secret_values() -> None:
    """
    Retrieve the API_TOKEN and the BOT_NAME from the pickled file

    :return: None
    """

    global API_TOKEN, BOT_NAME
    file = open("secret", "rb")
    secrets = pickle.load(file)
    API_TOKEN = secrets[API_TOKEN]
    BOT_NAME = secrets[BOT_NAME]
    file.close()


get_secret_values()
bot = telebot.TeleBot(API_TOKEN)


def setter(message, nxt_func, same_func, which="x", nxt_which="Y", normal=False):
    global n
    txt = message.text
    if txt.lower() == "stop":
        terminate_process(message)
    else:
        if "Z" in same_func.__name__:
            shp = x[-1].shape
            if len(shp) == 1:
                x[-1], y[-1] = np.meshgrid(x[-1], y[-1])
        code, msg = format_string_numpy_array(txt, which)
        if code:
            if "generate" not in nxt_func.__name__:
                rep = bot.send_message(message.chat.id, f"set {nxt_which}\nValues so far :\nX = {string_x}\nY = {string_y}")
                bot.register_next_step_handler(rep, nxt_func)
            else:
                if not normal:
                    code, msg = nxt_func()
                    if code:
                        bot.send_photo(message.chat.id, open(msg, "rb"))
                    else:
                        bot.send_message(message.chat.id, msg)
                    clear_list()
                else:
                    n -= 1
                    nxt_func(n, message)
        else:
            rep = bot.send_message(message.chat.id, msg)
            bot.register_next_step_handler(rep, same_func)


def format_string_numpy_array(string: str, which: str = "x") -> (bool, str):
    """
    This function is used to parse the string to numpy array and then
    append it to the respective list

    :param string: The string which is needed to be parsed to numpy array

    :param which: Which array it should append the parsed string

    :return: (bool,str) Returns True and empty string if there is no errors while parsing
            the string otherwise it return False and the error message
    """

    if re.search(r'\D', string.replace(",", "")):  # Check whether the given string contains alphabets
        try:
            if which == "x":
                x.append(eval(string))  # Evaluate the string and append the value to x
                string_x.append(string)
            elif which == "y":
                y.append(eval(string))  # Evaluate the string and append the value to y
                string_y.append(string)
                if x[-1].size != y[-1].size:
                    y.pop()
                    return False, f"Make sure the length of x and y are same\n" \
                                  f"Please provide the y value with length of {x[-1].size}"
                regex_format = re.sub("\[.*?\]", "", string)
                labels.append(regex_format.replace("**", "^").replace("np.", ""))

            elif which == "z":
                z.append(eval(string))  # Evaluate the string and append the value to y
                if x[-1].size != y[-1].size != z[-1].size:
                    z.pop()
                    return False, f"Make sure the length of x and y and z are same\n" \
                                  f"Please provide the z value with length of {x[-1].size}"
            return True, ""
        except Exception as e:
            return False, f"Please provide a correct syntax\n{new_line.join(e.args)}"

    else:
        try:
            if which == "x":
                x.append(np.array(list(map(float, string.split(",")))))
                string_x.append(string)
            elif which == "y":
                y.append(np.array(list(map(float, string.split(",")))))
                string_y.append(string)
                if x[-1].size != y[-1].size:
                    y.pop()
                    return False, f"Make sure the length of x and y are same\n" \
                                  f"Please provide the y value with length of {x[-1].size}"

            elif which == "z":
                z.append(np.array(list(map(float, string.split(",")))))
                if x[-1].size != y[-1].size != z[-1].size:
                    z.pop()
                    return False, f"Make sure the length of x and y and z are same\n" \
                                  f"Please provide the z value with length of {x[-1].size}"

            return True, ""
        except Exception as e:
            return False, f"Make sure you have given the \n{which.upper()} values in comma separated manner\nE.g 1,2," \
                          f"3,4,6,7 \n{new_line.join(e.args)}"


@bot.message_handler(commands=['start'])
def greet(message):
    bot.reply_to(message, "Hey! How is it going ? ")


@bot.message_handler(commands=['help', "h", "HELP", "Help"])
def helper(message):
    bot.send_message(message.chat.id, helper_txt)


@bot.message_handler(func=lambda message: True if "/" not in message.text else False)
def bot_response(message):
    if message.text.lower() == "bye":
        bot.reply_to(message, "Good bye! Have a nice day")
        return
    response = CHAT_BOT.get_response(message.text)
    bot.reply_to(message, response)


@bot.message_handler(commands=["graph"])
def graph(message):
    rep = bot.send_message(message.chat.id, f"How many plots do you want ?\nInput an integer")
    bot.register_next_step_handler(rep, get_n_plots)


@bot.message_handler(commands=["piegraph"])
def pie_chart(message):
    rep = bot.send_message(message.chat.id, f"Set the data")
    bot.register_next_step_handler(rep, set_pie)


@bot.message_handler(commands=["histgraph"])
def histogram(message):
    rep = bot.send_message(message.chat.id, f"Set the data")
    bot.register_next_step_handler(rep, set_hist)


@bot.message_handler(commands=["3dwireframe"])
def wireframe_plot(message):
    rep = bot.send_message(message.chat.id, f"Set X")
    bot.register_next_step_handler(rep, set_wire_X)


@bot.message_handler(commands=["3dcontour"])
def contour_plot(message):
    rep = bot.send_message(message.chat.id, "Set X")
    bot.register_next_step_handler(rep, set_cont_X)


@bot.message_handler(commands=["3dsurface"])
def surface_plot(message):
    rep = bot.send_message(message.chat.id, "Set X")
    bot.register_next_step_handler(rep, set_sur_X)


@bot.message_handler(commands=["scatter"])
def scatter_plot(message):
    rep = bot.send_message(message.chat.id, "Set X")
    bot.register_next_step_handler(rep, set_scat_X)


def get_n_plots(message):
    global n
    n = int(message.text)
    generate_nplots(n, message)


def generate_nplots(n_plots: int, message):
    if n_plots <= 0:
        code, msg = normal_graph()
        if code:
            bot.send_photo(message.chat.id, open(msg, "rb"))
        else:
            bot.send_message(message.chat.id, msg)
        clear_list()
    else:
        rep = bot.send_message(message.chat.id, f"Set X")
        bot.register_next_step_handler(rep, set_x)


def set_x(message):
    global x
    setter(message, nxt_func=set_y, same_func=set_x, which="x", nxt_which="Y")


def set_y(message):
    global y, n
    setter(message, nxt_func=generate_nplots, same_func=set_y, which="y", normal=True)


def terminate_process(message):
    bot.send_message(message.chat.id, "Instruction terminating ......")
    bot.send_message(message.chat.id, "Termination Success")
    clear_list()


def set_pie(message):
    setter(message, nxt_func=generate_pie_chart, same_func=set_pie, which="x")


def set_hist(message):
    setter(message, nxt_func=generate_histograms, same_func=set_hist, which="x")


def set_wire_X(message):
    global x
    setter(message, nxt_func=set_wire_Y, same_func=set_wire_X)


def set_wire_Y(message):
    global y
    setter(message, nxt_func=set_wire_Z, same_func=set_wire_Y, which="y", nxt_which="Z")


def set_wire_Z(message):
    global z
    setter(message, nxt_func=generate_3dwireframe, same_func=set_wire_Z, which="z")


def set_cont_X(message):
    global x
    setter(message, nxt_func=set_cont_Y, same_func=set_cont_X)


def set_cont_Y(message):
    global y
    setter(message, nxt_func=set_cont_Z, same_func=set_cont_Y, which="y", nxt_which="Z")


def set_cont_Z(message):
    global z
    setter(message, nxt_func=generate_3dcontour, same_func=set_cont_Z, which="z")


def set_sur_X(message):
    global x
    setter(message, nxt_func=set_sur_Y, same_func=set_sur_X)


def set_sur_Y(message):
    global y
    setter(message, nxt_func=set_sur_Z, same_func=set_sur_Y, which="y", nxt_which="Z")


def set_sur_Z(message):
    global z
    setter(message, nxt_func=generate_3dcontour, same_func=set_cont_Z, which="z")


def set_scat_X(message):
    global x
    setter(message, nxt_func=set_scat_Y, same_func=set_scat_X)


def set_scat_Y(message):
    global y
    setter(message, nxt_func=generate_scatter, same_func=set_scat_Y, which="y")


bot.polling()
