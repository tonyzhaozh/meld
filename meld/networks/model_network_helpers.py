import functools
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

########################################
# Priors and Posteriors
########################################


class ConstantMultivariateNormalDiag(tf.Module):
  """ Distribution used for modeling 1st-step prior of LVM """

  def __init__(self, latent_size, scale=None, name=None):
    super(ConstantMultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size
    self.scale = scale

  def __call__(self, *inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
    shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    if self.scale is None:
      scale_diag = tf.ones(shape)
    else:
      scale_diag = tf.ones(shape) * self.scale
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class MultivariateNormalDiag(tf.Module):
  """ Distribution used for modeling priors and posteriors of LVM """

  def __init__(self, base_depth, latent_size, scale=None, name=None):
    super(MultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(
        2 * latent_size if self.scale is None else latent_size)
    # scale here indicates whether to predict the covariance. If "scale" has content, the output is
    # of size latent_size which is the mean, and scale@scale.T is the covariance. If yes, the output predicts "scale",
    # output is thus 2x as long.

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., :self.latent_size] # the mean of distribution
    if self.scale is None:
      assert out.shape[-1].value == 2 * self.latent_size
      scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5
    else:
      assert out.shape[-1].value == self.latent_size
      scale_diag = tf.ones_like(loc) * self.scale # here indicates scale is a scalar, thus same variance on each dim.
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


########################################
# Decoders
########################################


class ImageDecoder(tf.Module):
  """ Probabilistic decoder for `p(x_t | z_t)` """

  def __init__(self, base_depth, channels=3, scale=1.0, name=None, double_camera=False):
    super(ImageDecoder, self).__init__(name=name)
    self.scale = scale
    self.double_camera = double_camera # if not square: 2*1 rectangle
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)

    if not self.double_camera:
      self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
      self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
      self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
      self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
      self.conv_transpose5 = conv_transpose(channels, 5, 2, activation=None)
    else:
      self.conv_transpose1 = conv_transpose(8 * base_depth, (8, 4), padding="VALID")
      self.conv_transpose2 = conv_transpose(4 * base_depth, (6, 3), 2)
      self.conv_transpose3 = conv_transpose(2 * base_depth, (6, 3), 2)
      self.conv_transpose4 = conv_transpose(base_depth, (6, 3), 2)
      self.conv_transpose5 = conv_transpose(channels, (10, 5), 2, activation=None)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)

class ImageDecoderState(tf.Module):
  """ Normal, Decoder for  `p(s_t | z_t)` """

  def __init__(self, base_depth, state_size, scale=None, name=None):
    super(ImageDecoderState, self).__init__(name=name)
    self.state_size = state_size
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 * state_size if self.scale is None else state_size)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., :self.state_size]

    if self.scale is None:
      assert out.shape[-1].value == 2 * self.state_size
      scale_diag = tf.nn.softplus(out[..., self.state_size:]) + 1e-5
    else:
      assert out.shape[-1].value == self.state_size
      scale_diag = tf.ones_like(loc) * self.scale # here indicates scale is a scalar, thus same variance on each dim.

    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class RewardDecoder(tf.Module):
  """ Normal, Decoder for  `p(r_t | z_t)` """

  def __init__(self, base_depth, scale=None, name=None):
    super(RewardDecoder, self).__init__(name=name)
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 if self.scale is None else 1)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., 0]
    if self.scale is None:
      assert out.shape[-1].value == 2
      scale = tf.nn.softplus(out[..., 1]) + 1e-5
    else:
      assert out.shape[-1].value == 1
      scale = self.scale
    return tfd.Normal(loc=loc, scale=scale)


class BiggerRewardDecoder(tf.Module):
  """ Normal, Decoder for  `p(r_t | z_t)` """

  def __init__(self, base_depth, scale=None, name=None):
    super(BiggerRewardDecoder, self).__init__(name=name)
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(2*base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense3 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense4 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 if self.scale is None else 1)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.dense3(out)
    out = self.dense4(out)
    out = self.output_layer(out)
    loc = out[..., 0]
    if self.scale is None:
      assert out.shape[-1].value == 2
      scale = tf.nn.softplus(out[..., 1]) + 1e-5
    else:
      assert out.shape[-1].value == 1
      scale = self.scale
    return tfd.Normal(loc=loc, scale=scale)


class DiscountDecoder(tf.Module):
  """ Bernoulli, Decoder for  `p(d_t | z_{t+1})` """

  def __init__(self, base_depth, name=None):
    super(DiscountDecoder, self).__init__(name=name)
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(1)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    logits = tf.squeeze(out, axis=-1)
    return tfd.Bernoulli(logits=logits)


########################################
# Compressor
########################################


class ModelCompressor(tf.Module):
  """Feature extractor: images-->features """

  def __init__(self, base_depth, feature_size, name=None, double_camera=False):
    super(ModelCompressor, self).__init__(name=name)
    self.base_depth = base_depth
    self.double_camera = double_camera # if double_camera: 2*1 rectangle
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    if not self.double_camera:
      self.conv1 = conv(base_depth, 5, 2)
      self.conv2 = conv(2 * base_depth, 3, 2)
      self.conv3 = conv(4 * base_depth, 3, 2)
      self.conv4 = conv(8 * base_depth, 3, 2)
      self.conv5 = conv(8 * base_depth, 4, padding="VALID")
    else:
      self.conv1 = conv(base_depth, (10, 5), 2)  # conv: filters, kernel_size, stride
      self.conv2 = conv(2 * base_depth, (6, 3), 2)
      self.conv3 = conv(4 * base_depth, (6, 3), 2)
      self.conv4 = conv(8 * base_depth, (6, 3), 2)
      self.conv5 = conv(8 * base_depth, (8, 4), padding="VALID")

  def __call__(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)


class ModelCompressorState(tf.Module):
  """Feature extractor: states-->features """

  def __init__(self, base_depth, feature_size, name=None):
    super(ModelCompressorState, self).__init__(name=name)
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(feature_size)

  def __call__(self, *inputs):

    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs

    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    return out