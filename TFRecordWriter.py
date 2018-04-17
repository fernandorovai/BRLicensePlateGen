import tensorflow as tf

class TFExample:
    def __init__(self):
        self.height             = None  # Image height
        self.width              = None  # Image width
        self.filename           = None  # Filename of the image. Empty if image is not from file
        self.sourceID           = None  # Intern ID
        self.encodedImageData   = None  # Encoded image bytes
        self.imageFormat        = None  # b'jpeg' or b'png'
        self.xMins              = []    # List of normalized left x coordinates in bounding box (1 per box)
        self.xMaxs              = []    # List of normalized right x coordinates in bounding box (1 per box)
        self.yMins              = []    # List of normalized top y coordinates in bounding box (1 per box)
        self.yMaxs              = []    # List of normalized bottom y coordinates in bounding box (1 per box)
        self.classesText        = []    # List of string class name of bounding box (1 per box)
        self.classes            = []    # List of integer class id of bounding box (1 per box)
        self.tilesID            = []    # List of tiles from certain image

class TFRecordWriter:
    def __init__(self, tfrecordsFilename):
        self._writer  = tf.python_io.TFRecordWriter(tfrecordsFilename)

    def createTfExample(self, tfExample):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height':             self.int64_feature(tfExample.height),
            'image/width':              self.int64_feature(tfExample.width),
            'image/filename':           self.bytes_feature(tfExample.filename),
            'image/source_id':          self.bytes_feature(tfExample.sourceID),
            'image/encoded':            self.bytes_feature(tfExample.encodedImageData),
            'image/format':             self.bytes_feature(tfExample.imageFormat),
            'image/object/bbox/xmin':   self.float_list_feature(tfExample.xMins),
            'image/object/bbox/xmax':   self.float_list_feature(tfExample.xMaxs),
            'image/object/bbox/ymin':   self.float_list_feature(tfExample.yMins),
            'image/object/bbox/ymax':   self.float_list_feature(tfExample.yMaxs),
            'image/object/class/text':  self.bytes_list_feature(tfExample.classesText),
            'image/object/class/label': self.int64_list_feature(tfExample.classes)
        }))
        return tf_example

    def closeTfStream(self):
        self._writer.close()

    def appendExampleToTfStream(self, parsedTfExample):
        self._writer.write(parsedTfExample.SerializeToString())

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))