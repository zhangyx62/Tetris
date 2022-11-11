import argparse
import torch
import cv2
from tetris import Tetris


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    torch.manual_seed(123)
    model = torch.load('trained_models/tetris_10000', map_location=torch.device('cpu')).to(device)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    model = model.to(device)
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5 * opt.width * opt.block_size), opt.height * opt.block_size))
    while True:
        state = env.get_simple_image().to(device)
        print('----------')
        for y in range(state.shape[1]):
            for x in range(state.shape[2]):
                if state[0, y, x] == 0:
                    print(' ', end='')
                else:
                    print('#', end='')
            print()
        print('----------')
        with torch.no_grad():
            predictions = model(state[None, :])[0]
        action = torch.argmax(predictions).item()
        reward, done, next_state = env.step((action // 4, action % 4), render=True, video=out)

        if done:
            out.release()
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
