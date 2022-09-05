import argparse
import os
import yaml
from modules.visualizer import Visualizer
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--point_cloud_view', '-pcv',
        type=str,
        required=False,
        default=False,
        help='Parameter to see the semantic point cloud, default False',
    )

    parser.add_argument(
        '--median_filter', '-med',
        type=str,
        required=False,
        default=False,
        help='Applying median filter , default False',
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config\\labels\\semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default=None,
        required=False,
        help='Alternate location for labels, to use predictions folder. '
             'Must point to directory containing the predictions in the proper format '
             ' (see readme)'
             'Defaults to %(default)s',
    )

    parser.add_argument(
        '--width', '-w',
        type=str,
        default=1024,
        required=False,
        help='Width of the projected image'
             ' (see readme)'
             'Defaults to %(default)s',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Sequence", FLAGS.sequence)
    print("Predictions", FLAGS.predictions)
    print("Point cloud viewer", FLAGS.point_cloud_view)
    print("Median filter", FLAGS.median_filter)
    print("Width", FLAGS.width)
    print("*" * 80)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # does sequence folder exist?
    scan_label = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "labels")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()
    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_label)) for f in fn]
    label_names.sort()

    prediction_names=None
    if FLAGS.predictions is not None:
        prediction_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(FLAGS.predictions)) for f in fn]
        prediction_names.sort()

        if prediction_names == []:
            print("Wrong path to predictions!")

    med_filter = False
    if FLAGS.median_filter:
        med_filter = True

    w = 1024
    try:
        if type(int(FLAGS.width)) == int:
            w = int(FLAGS.width)
    except:
        print("For parameter -w you need to type a number! Will be used the default value 1024!")

    vis = Visualizer(64, w, scan_names, label_names,prediction_names, med_filter)

    # run the visualizer
    if FLAGS.point_cloud_view:
        vis.vizualize_pcd()
    else:
        vis.visualize_projections()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

