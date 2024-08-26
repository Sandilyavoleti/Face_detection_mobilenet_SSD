import tensorflow as tf

class SSDLoss:
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 beta=1.0):
        self.neg_pos_ratio = tf.constant(neg_pos_ratio, dtype=tf.int32)
        self.n_neg_min = tf.constant(n_neg_min, dtype=tf.int32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)

    def smooth_L1_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * tf.square(y_true - y_pred)
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-15)
        log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]

        classification_loss = self.log_loss(y_true[:, :, :2], y_pred[:, :, :2])

        localization_loss = self.smooth_L1_loss(y_true[:, :, -4:], y_pred[:, :, -4:])

        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-12], axis=-1)

        n_positive = tf.reduce_sum(positives)

        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)

        neg_class_loss_all = classification_loss * negatives
        n_neg_losses = tf.reduce_sum(tf.cast(neg_class_loss_all > 0, tf.int32))

        n_negative_keep = tf.minimum(
            tf.maximum(self.neg_pos_ratio * tf.cast(n_positive, dtype=tf.int32), self.n_neg_min),
            n_neg_losses)

        def no_negatives():
            return tf.zeros([batch_size], dtype=tf.float32)

        def has_negatives():
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, k=n_negative_keep, sorted=False)
            negatives_keep = tf.scatter_nd(
                tf.expand_dims(indices, axis=1),
                updates=tf.ones_like(indices, dtype=tf.float32),
                shape=tf.shape(neg_class_loss_all_1D)
            )
            negatives_keep = tf.reshape(negatives_keep, [batch_size, n_boxes])
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, 0), no_negatives, has_negatives)

        class_loss = pos_class_loss + neg_class_loss
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        total_loss = (self.beta * class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)

        return total_loss