# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
""" Session Dataset
    - Class responsible for retrieving outputs by feeding data to a model running in Tensorflow session
"""
import logging

import numpy as np
from tornado import gen

from diplomacy_research.models.datasets.base_builder import VarProtoField
from diplomacy_research.models.datasets.feedable_dataset import FeedableDataset
from diplomacy_research.utils.tensorflow import tf

# Constants
LOGGER = logging.getLogger(__name__)

class SessionDataset(FeedableDataset):
    """ This object is responsible for retrieving model outputs from Tensorflow session """

    def __init__(self, model_dir, signature, dataset_builder, cluster_config=None):
        """ Constructor
            :param model_dir: Saved model directory containing SavedModel.pb file.
            :param signature: The output of adapter.get_signature() - signature of all the possible calls
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param cluster_config: Optional. If set, the cluster configuration will be used for distributed training.
        """
        # pylint: disable=too-many-arguments
        super(SessionDataset, self).__init__(dataset_builder=dataset_builder,
                                          cluster_config=cluster_config)
        self.model_dir = model_dir
        self.signature = signature

        # Building the dataset
        tf.load_op_library('./seeded_random.so')
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, ["serve"], self.model_dir)

        self.build()

    @property
    def can_support_iterator(self):
        """ Determines if the dataset can support an iterator or if it is a remote (RPC) dataset """
        return False

    def build(self):
        """ Builds default features """
        assert 'request_id' in self.proto_fields, 'You need to have a "request_id" field.'

        # Padding output shapes with None
        output_types = self.dataset_builder.output_types
        output_shapes = self.dataset_builder.output_shapes
        output_shapes = {key: [None] + list(shape) for key, shape in output_shapes.items()}

        # Building a list of generic default values from the output types and output shapes
        for feature_name, feature_shape in output_shapes.items():
            if output_types[feature_name] == np.object:
                self.default_features[feature_name] = np.array(bytes('', 'utf-8'), dtype=np.object).reshape(1)
            elif isinstance(self.proto_fields[feature_name], VarProtoField):
                self.default_features[feature_name] = np.array([], dtype=output_types[feature_name]).reshape(1, 0)
            else:
                self.default_features[feature_name] = np.zeros([1] + feature_shape[1:], dtype=output_types[feature_name])

    def start(self, session):
        """ Starts the dataset
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        self._is_started = True

    def initialize(self, session):
        """ Initializes the dataset (and its iterator)
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        self._is_initialized = True

    def has_queue(self, queue_name):
        """ Determines if the feedable dataset already has a queue with the specified name """
        return queue_name in self.signature

    @gen.coroutine
    def get_results(self, queue_name, item, **kwargs):
        """ Computes the outputs of a name using item as inout
            :param queue_name: The name of the queue where to put the item (or model_name/queue_name)
            :param item: A dictionary with the fields required for that queue
            :return: A tornado.concurrent.Future that will be set with the results when they become available
        """

        request = {}
        feed_dict = {}

        input_tensor_names = {
            'request_id': "import/IteratorGetNext:12",
            'decoder_type': "import/decoder_type:0",
            'player_seed': "import/IteratorGetNext:10",
            'board_state': "import/IteratorGetNext:1",
            'board_alignments': "import/IteratorGetNext:0",
            'prev_orders_state': "import/IteratorGetNext:11",
            'decoder_inputs': "import/IteratorGetNext:5",
            'decoder_lengths': "import/IteratorGetNext:6",
            'candidates': "import/IteratorGetNext:2",
            'noise': "import/IteratorGetNext:9",
            'temperature': "import/IteratorGetNext:13",
            'dropout_rate': "import/IteratorGetNext:8",
            'current_power': "import/IteratorGetNext:3",
            'current_season': "import/IteratorGetNext:4",
            'draw_target': "import/IteratorGetNext:7",
            'value_target': "import/IteratorGetNext:14"
        }

        if not self.has_queue(queue_name):
            LOGGER.warning('The method "%s" could not be found.', queue_name)
            return None
        if not isinstance(item, dict):
            LOGGER.warning('The item object passed to get_results must be a dictionary.')
            return None

        # Preparing the item
        item['request_id'] = bytes('', 'utf-8')
        item = self.prepare_item(item)

        # Setting the keys in items
        # Adding a leading batch dimension, so that TF Serving can batch items properly
        # i.e. (dim_1, dim_2) --> (1, dim_1, dim_2)
        for key in item:
            batched_item_key = item[key][None, ...] if isinstance(item[key], np.ndarray) else [item[key]]
            request[key] = np.array(batched_item_key, dtype=self.dataset_builder.output_types[key])

        # Setting the placeholders defined in the signature
        placeholders = self.signature[queue_name].get('placeholders', {})
        for ph_name in placeholders:
            ph_value, ph_dtype = placeholders[ph_name]
            request[ph_name] = np.array(ph_value, dtype=ph_dtype)

        # Adding generic default values (all zeros)
        for default_key, default_val in self.default_features.items():
            if default_key not in item:
                request[default_key] = default_val

        for key, val in request.items():
            feed_dict[input_tensor_names[key]] = val

        if queue_name == "policy_evaluate":
            output_tensors = ["import/policy_1/decoder_scope/Cast_3:0", "import/policy_1/decoder_scope/mul_8:0"]
        elif queue_name == "policy_beam_search":
            output_tensors = ["import/transpose:0", "import/policy_1/decoder_scope/decoder_1/while/Exit_12:0"]

        results = self.sess.run(output_tensors, feed_dict=feed_dict)
        results = [output[0, ...] for output in results]
        
        return results

    def close(self):
        """ Closes the underlying channel """
        super(SessionDataset, self).close()
        if self.sess:
            self.sess.close()
            self.sess = None
