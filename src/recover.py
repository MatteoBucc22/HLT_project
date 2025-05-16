import os
import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riprendi training da checkpoint LoRA")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Percorso alla cartella adapter per ricominciare")
    parser.add_argument("--start_epoch", type=int, required=True,
                        help="Numero dell'epoca da cui ripartire")
    args = parser.parse_args()

    # Avvia il training riprendendo dal checkpoint indicato
    train(resume_from=args.checkpoint_dir, start_epoch=args.start_epoch)
