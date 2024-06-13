
import argparse
import os

import data.fileIO as fileIO
import yaml
from algorithm.geneticAlgorithm import Genetic

# -------------------------------------------------------------
# Arguments


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')

    # Required params
    p.add_argument("-binPath", 
        help="Path to the sequences folder of LiDAR scan bins", 
        nargs='?', const="",
        default="")

    p.add_argument("-labelPath", 
        help="Path to the sequences label files", 
        nargs='?', const="",
        default="")

    p.add_argument("-predPath", 
        help="Path to the prediction label files created by the models", 
        nargs='?', const="",
        default="")
    
    p.add_argument("-modifidePrediction", 
        help="Path to the modifide prediction label files created by mutations", 
        nargs='?', const="",
        default="")
    
    p.add_argument("-mdb", 
        help="Path to the connection string for mongo", 
        nargs='?', const="",
        default="")

    p.add_argument("-modelDir", 
        help="Path to the directory where the models are saved", 
        nargs='?', const="",
        default="")


    p.add_argument('-models', 
        help='Models (SUTs) to evaluate comma seperated: cyl,spv,js3c_gpu,sal,sq3',
        nargs='?', const="", default="")

    p.add_argument('-baseSequence', 
        help='Base sequences number to perform mutation',
        nargs='?', const="", default="")

    # Tool configurable params
    p.add_argument("-saveAt", 
        help="Location to save the tool output",
        nargs='?', const=os.getcwd(), 
        default=os.getcwd())

    p.add_argument('-algorithms', 
    help='Algorithms to search for positions and objects to insert in lidar, that make models error',
    nargs='?', const="", default="Genetic Algorithm")

    p.add_argument('-pk', 
        help='Use prior knowledge of the number and quantity of instances of the original point cloud data',
        action='store_true', default=False)
    
    p.add_argument('-IfInstance', 
        help='Genetic Algorithm includes instances',
        action='store_true', default=True)
    
    p.add_argument('-IfWeather', 
        help='Genetic Algorithm includes weathers',
        action='store_true', default=True)

    p.add_argument(
      '-config',
      type=str,
      required=False,
      default="config.yaml",
      help='Dataset config file. Defaults to %(default)s')
    
    # Optional Flags
    p.add_argument('-vis', 
        help='Visualize with Open3D',
        action='store_true', default=False)

    p.add_argument("-assetId", 
        help="Asset Identifier, optional forces the tool to choose one specific asset", 
        nargs='?', const=None, default=None)
    p.add_argument("-sequence", 
        help="Sequences number, provide as 00 CANNOT BE USED WITHOUT scene backdoor to force add to choose base scene", 
        nargs='?', const=None, default=None)
    p.add_argument( "-scene", 
        help="Specific scene number provide full ie 002732, CANNOT BE USED WITHOUT sequence",
        nargs='?', const=None, default=None)
    
    return p.parse_args()

    


# ----------------------------------------------------------

def main():

    print("\n\n------------------------------")
    print("\n\nStarting LiDAR Test Generation\n\n")
    
    # Get arguments 
    args = parse_args()
    algorithm = args.algorithms

    if algorithm == "Genetic Algorithm":
        GA = Genetic(args)
        GA.run()
    elif algorithm == "semlidarfuzz":
        GA = Genetic(args)
        GA.random_all()

if __name__ == '__main__':
    main()



