%  predict       - �������Բ��Լ��ķ�����
%  ground_truth - ���Լ�����ȷ��ǩ,����ֻ���Ƕ����࣬��0��1
%  auc            - ����ROC���ߵ������µ����
function auc = plot_roc( predict, ground_truth )
%��ʼ��Ϊ��1.0, 1.0��
%�����ground_truth������������Ŀpos_num�͸���������Ŀneg_num

pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==-1);

m=size(ground_truth,1);
[pre,Index]=sort(predict);
ground_truth=ground_truth(Index);
x=zeros(m+1,1);
y=zeros(m+1,1);
auc=0;
x(1)=1;y(1)=1;

for i=2:m
 TP=sum(ground_truth(i:m)==1);FP=sum(ground_truth(i:m)==-1);
 x(i)=FP/neg_num;
 y(i)=TP/pos_num;
 auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
end;

 x(m+1)=0;y(m+1)=0;
 auc=auc+y(m)*x(m)/2;
 plot(x,y);
 hold on
 plot([0,1],[1,0],':');
 grid on
 xlabel('FPR','FontSize',12);
 ylabel('TPR','FontSize',12);
 title('ROC space','FontSize',12);
 hold off
end
