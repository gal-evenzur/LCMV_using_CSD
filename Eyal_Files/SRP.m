function [DOAs_out ] = SRP(mics,mics_pos,mics_setup,fs,Noise_only, window,Num_of_speakers,reso,DOAs)
Number_Of_Mics = sum(  mics_setup  );
c = 340;
%%
%%%%%%%
mics_pos = mics_pos';
mics_pos = mics_pos(:,find( mics_setup));
mics = mics(:,:,find(mics_setup));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Searched_DOAs = [5:reso:185-reso ];
for j = 1:Number_Of_Mics
    for DOA = 0:359
        doa_unit_vec = [cosd(DOA) sind(DOA)];
        mics_inter_vec = mics_pos(:,j) - mics_pos(:,1);
        mics_inter_tdoa = - (doa_unit_vec * mics_inter_vec) / c;
        mics_inter_tdoa_1(j,DOA+1) = 2*pi/window*mics_inter_tdoa*fs;
    end
end
alpha = 0.7;
frame_before = 8;
frame_after = 5;
L1 = floor(100/fs*window);
L2 = floor(3500/fs*window);
for n=1:Number_Of_Mics
        for m=n+1:Number_Of_Mics  
            for k = L1:L2
                for d = 1:length(Searched_DOAs)
                    expected_phase(n,m,k,d) = exp(-sqrt(-1)*(mics_inter_tdoa_1(m,Searched_DOAs(d)+1) - mics_inter_tdoa_1(n,Searched_DOAs(d)+1))*k);
                end
            end
        
        end
end
%%%%
DOAs_out = zeros(length(mics(1,:,1))-frame_before-frame_after,1);
G12 = zeros(window/2+1,Number_Of_Mics,Number_Of_Mics);
R = zeros(window/2+1,length(Searched_DOAs));
index_frame = 1;
win_vad = hamming(21);
for x = frame_before+1 : 1 : length(mics(1,:,1))-frame_after
%%%%%%% SRP - PHAT
    Mic = mics(:,x-frame_before:x+frame_after,:);
    Mic = permute(Mic,[3 2 1]);
    R(1:window/2+1,1:length(Searched_DOAs))=0;
    for n=1:Number_Of_Mics
        for m=n+1:Number_Of_Mics 
            for pp =1:window/2+1
                Mic_temp = Mic(n,:,pp);
                Mic_temp = Mic_temp.*win_vad(10-frame_before:10+frame_after).';
                G12(pp,n,m) = alpha*G12(pp,n,m) + (1-alpha)*Mic_temp*Mic(m,:,pp)';
            end
            G = G12(:,n,m);
            for k = L1:L2
                for d = 1:length(Searched_DOAs)
                    R(k,d) = R(k,d) + real(expected_phase(n,m,k,d)*G(k));
                end
            end          
        end
    end
    [~, observations] = max(sum(R(L1:L2 ,:),1));
    DOAs_out (index_frame) =  observations ;
    index_frame = index_frame+1;
end
