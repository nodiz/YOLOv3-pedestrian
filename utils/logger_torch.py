from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision


class Logger(object):
    def __init__(self, log_dir, active):
        """Create a summary writer logging to log_dir."""
        now = datetime.datetime.now()
        dt_string = now.strftime("%d%H%M")
        self.writer = SummaryWriter(log_dir, comment=dt_string)
        self.active = active

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if not self.active:
            return
        self.writer.add_scalar(tag, value, global_step=step)
        self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        if not self.active:
            return

        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, global_step=step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        if not self.active:
            return
        self.writer.add_image(tag, image, global_step=step)
        self.writer.flush()

    def list_of_images(self, tag, images, step):
        if not self.active:
            return
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, grid, global_step=step)
        self.writer.flush()

