# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf
import numpy as np
import scipy.misc
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x


class Logger(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = tf.summary.create_file_writer(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    with self.writer.as_default():
      tf.summary.scalar(tag, value,step)
      self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""

    img_summaries = []
    for i, img in enumerate(images):
      # Write the image to a string
      try:
        s = StringIO()
      except:
        s = BytesIO()
      scipy.misc.toimage(img).save(s, format="png")

      # Create an Image object
      img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                 height=img.shape[0],
                                 width=img.shape[1])
      # Create a Summary value
      img_summaries.append(tf.Summary.Value(
          tag='%s/%d' % (tag, i), image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=img_summaries)
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create and write Summary
    with self.writer.as_default():
      tf.summary.histogram(tag, values, step)
      self.writer.flush()
    







