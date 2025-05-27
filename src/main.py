from roBERTa_qqp.train_roBERTa_qqp import train as train_roberta_qqp
from roBERTa_mrpc.train_roBERTa_mrpc import train as train_roberta_mrpc
from miniLM_qqp.train_miniLM_qqp import train as train_minilm_qqp
from miniLM_mrpc.train_miniLM_mrpc import train as train_minilm_mrpc



def main():

    print("Inizio training roBERTa su MRPC...")
    train_roberta_mrpc()

    print("Inizio training MiniLM su QQP...")
    train_minilm_qqp()

    print("Inizio training MiniLM su MRPC...")
    train_minilm_mrpc()

    print("Inizio training roBERTa su QQP...")
    train_roberta_qqp()

if __name__ == "__main__":
    main()
