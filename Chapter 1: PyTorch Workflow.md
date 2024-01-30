# **Pytorch Workflow**
flowchart LR
    A(Get the data ready, data is in the tensor) -->B(Build or pick a pretrain model)
    B --Pick a loss function and optimizer, Build a training loop-->C(Fit the model to the data)
    C --make a prediction-->D(Evaluate the model)
    D --> E(Improve through experimentation)
    E --> F(Save and reloaded our trained model)
