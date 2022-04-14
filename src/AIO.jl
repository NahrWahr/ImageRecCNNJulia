using Statistics
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition
using CUDA

trainX, trainY = CIFAR10.traindata(Float32)
labels = onehotbatch(trainY, 0:9)

train = ([(trainX[:,:,:,i], labels[:,i]) for i in partition(1:49000, 1000)]) |> gpu
valset = 49001:50000

valX = trainX[:,:,:,valset] |> gpu
valY = labels[:, valset] |> gpu

m = Chain(
	  Conv((5,5), 3=>16, relu),
	  MaxPool((2,2)),
	  Conv((5,5), 16=>8, relu),
	  MaxPool((2,2)),
	  x -> reshape(x, :, size(x, 4)),
	  Dense(200, 120),
	  Dense(120, 84),
	  Dense(84,10),
	 softmax) |> gpu

using Flux: crossentropy, Momentum

loss(x, y) = sum(crossentropy(m(x), y))
opt = Momentum(0.01)

accuracy(x, y) = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))

epochs = 10

for epoch = 1:epochs
	for d in train
		gs = gradient(params(m)) do
			l = loss(d...)
		end
		update!(opt, params(m), gs)
	end
	@show accuracy(valX, valY)
end

testX, testY = CIFAR.testdata(Float32)
testLabels = onehotbatch(testY, 0:9)

test = gpu.([(testX[:,:,:,i], testLabels[:,i]) for i in partition(1:10000, 1000)])

plot(image(testX[:,:,:,rand(1:end)]))

ids = rand(1:10000, 5)
randTest = testX[:,:,:,ids] |> gpu
randTruth = testY[ids]
m(randTest)

accuracy(test[1]...)

classCorrect = zeros(10)
classTotal = zeros(10)

for i = 1:10
	preds = m(test[i][1])
	lab = test[i][2]
	for j = 1:1000
		predClass = findmax(preds[:, j])[2]
		actualClass = findmax(lab[:, j])[2]
		if predClass == actualClass
			classCorrect[predClass] +=1
		end
		classTotal[actualClass] +=1
	end
end
classCorrect ./ classTotal
