function [] = LLP(datasetName,number,d)  
    addpath('./common');
    [img,gt] = get_data(datasetName);
    img = img./max(img(:));
    class_num = length(unique(gt));
    [m,n,c] = size(img);  
    I = img_pca(img,1);
    I = padarray(I,[d,d],'replicate');
    resultI = zeros(m,n);  
    width = 2*d+1;  
    S = zeros(width,width);%pix Similarity  
    col_rol_index = reshape([1:(m+2*d)*(n+2*d)],m+2*d,n+2*d);
    dex = [1,11,26,33,52,58,85,104,111,121];
    B = cell(2,class_num-1);
    link_graph = reshape(1:m*n,m,n);
    pad_graph = padarray(link_graph,[d d],0);
    tic;
    for s = 1:size(number,2)
        nTrEachClass = number(s);
        for index=1:10
            tic;
            pad_img = padarray(img,[d,d],'replicate');
            pad_zero = padarray(ones(m,n),[d,d],0);
            [~, ~, ~, ~,index_train, index_test, ~, ~, ~, ~ ] = ChooseSample(img,gt,nTrEachClass, dex(index));
            index_test = find(gt(:)>0);
            index_test = setdiff(index_test,index_train);

            temp_i = zeros(m*n,1);
            temp_j = zeros(m*n,1);
            temp_v = zeros(m*n,1);
            
            label = zeros(m*n,class_num);
            for ll = 1:length(index_train)
                label(index_train(ll),gt(index_train(ll))+1) = 1;
            end
 
            color_map=[140 67 46;0 0 255;255 100 0;0 255 123;56 94 15;138 54 15;255 240 230;255 192 203;0 199 140;164 75 155;101 174 255;118 254 172;255 153 18; 60 91 112;255,255,0;255 255 125;255 0 255;100 0 255;0 172 254;0 255 0;171 175 80;101 193 60];
            ind = 1;

            [row, col] = ind2sub([m,n], setdiff([1:m*n],index_train));
            % 初始化临时变量数组
            temp_i_cell = cell(1, size(row,2));
            temp_j_cell = cell(1, size(row,2));
            temp_v_cell = cell(1, size(row,2));
            len_array = zeros(1, size(row,2));
            
            parfor i = 1:size(row,2)
                row_i = row(i);
                col_j = col(i);

                % 进行计算...
                temp = sum(abs(pad_img(row_i:row_i+2*d,col_j:col_j+2*d,:)-pad_img(row_i+d,col_j+d,:))./(abs(pad_img(row_i:row_i+2*d,col_j:col_j+2*d,:)-pad_img(row_i+d,col_j+d,:))+abs(pad_img(row_i:row_i+2*d,col_j:col_j+2*d,:)+pad_img(row_i+d,col_j+d,:))+eps),3);
                temp_div = temp(d:d+2,d:d+2);
                temp_div = temp_div([1:4,6:9]);
                temp = temp-min(temp_div);
                S = exp(-temp);
                S(int32(end/2),int32(end/2)) = 0;
                S = S(pad_zero(row_i:row_i+2*d,col_j:col_j+2*d)>0);
                S = S./sum(S(:));
                len = numel(S);

                mat_ind = pad_graph(row_i:row_i+2*d,col_j:col_j+2*d);
                mat_ind = mat_ind(mat_ind>0);

                % 存储每次迭代的结果
                temp_i_cell{i} = repmat(link_graph(row_i,col_j), len, 1);
                temp_j_cell{i} = mat_ind;
                temp_v_cell{i} = S(:);
                len_array(i) = len;
            end

            % 初始化合并后的变量
            total_len = sum(len_array);
            temp_i = zeros(total_len, 1);
            temp_j = zeros(total_len, 1);
            temp_v = zeros(total_len, 1);

            % 合并结果
            ind = 1;
            for i = 1:length(temp_i_cell)
                len = len_array(i);
                temp_i(ind:ind+len-1) = temp_i_cell{i};
                temp_j(ind:ind+len-1) = temp_j_cell{i};
                temp_v(ind:ind+len-1) = temp_v_cell{i};
                ind = ind + len;
            end

            % 处理index_train部分
            for i = 1:numel(index_train)
                temp_i(ind) = index_train(i);
                temp_j(ind) = index_train(i);
                temp_v(ind) = 1;
                ind = ind + 1;
            end

            % 创建稀疏矩阵
            W = sparse(temp_i(1:ind-1), temp_j(1:ind-1), temp_v(1:ind-1), m*n, m*n);
            old_predict = zeros(m,n);
            iter = 0;
            while true
                iter = iter + 1
                label = W*label;  
                [~,P] = max(label,[],2);
                predict = P-1;
                predict = reshape(predict,m,n);

                if(min(predict(:)) > 0)
                    [acc,~,~,~] = confusion(gt(index_test),predict(index_test))
                    sum(sum(predict ~= old_predict))
                    diff_p = sum(sum(predict ~= old_predict))*1.0/(m*n);
                    if diff_p < 0.001 || iter ==300
                        break;
                    end
                end

                old_predict = predict;
            end  
            [~,P] = max(label,[],2);
            predict = P-1;
            predict = reshape(predict,m,n);
            toc
            disp(['运行时间: ',num2str(toc)]);
            [oa(index),aa(index),kappa(index),ua,~]=confusion(gt(index_test),predict(index_test))
        end
        oa_mean = roundn(mean(oa),-4)
        oa_std = roundn(std(oa),-4)
        aa_mean = roundn(mean(aa),-4);
        aa_std = roundn(std(aa),-4);
        kappa_mean = roundn(mean(kappa),-4);
        kappa_std = roundn(std(kappa),-4);
    end  
    try
    	delete(gcp('nocreate'));
    catch ME
        disp('An error occurred:');
        disp(ME.message);
        delete(gcp('nocreate')); 
        fprintf('Error in function: %s\n', ME.stack(1).name);
        fprintf('Error message: %s\n', ME.message);
    end

end
