function rate = KNN_Classfier(train, trainlabel, test,true_testlabel, K)
%K Nearest Neighbor Classfier
%train, test: each column is a data
%trainlabel, true_testlabel: row vectors containing the labels

     aa=sum(train.*train,1); bb=sum(test.*test,1); ab=train'*test; 
     dist = sqrt(repmat(aa',[ 1 numel(bb)]) + repmat(bb, [numel(aa) 1]) - 2*ab);
     
     dist = real(dist); 

     [B,IX] = sort(dist);
     
     testlabel = zeros(1,size(test,2));
    
     for i  = 1:size(test,2)
          minindex = IX(1:K,i);
          
          neighborlabels = trainlabel(minindex);
          
          class_ids = unique(neighborlabels);
          C = numel(class_ids);
          neighbor_nums_per_class = zeros(1,C);
          for j = 1:C
              neighbor_nums_per_class(j) = sum(neighborlabels == class_ids(j));
          end
          [temp,id_id] = max(neighbor_nums_per_class);
          if sum(neighbor_nums_per_class==temp)==1          
              testlabel(i) = class_ids(id_id);
          else
              testlabel(i)=trainlabel(IX(1,i)); 
          end
     end
     rate=(sum(testlabel==true_testlabel)/length(true_testlabel))*100;

end 