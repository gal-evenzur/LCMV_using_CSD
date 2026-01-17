function [s_first,label_first,s_second,label_second,s_noise,r]=create_locations_18_dynamic(L,R,noise_R,num_jumps)
    load handel.mat
    addpath('C:\project\fillline.m');

    %% variable
    
    room_x = L(1);
    room_y = L(2);
    high=1;
    angle=360;
    radius_mics = 0.1;
    distance_from_woll=0.5;

    %% create circle & line & mic location
    
    distance_total=R+distance_from_woll+noise_R;
    end_point_x=room_x-(R+distance_from_woll+noise_R);
    end_point_y=room_y-(R+distance_from_woll+noise_R);
    Radius_X = (end_point_x-distance_total).*rand + distance_total;
    Radius_Y = (end_point_y-distance_total).*rand + distance_total;

    R_angle=randi([1,angle]); % take  rand 180 degrees of circle to create rand orientation of Microphone array
    t = linspace(-pi,2*pi,angle+angle/2); % 3 times 180 to create option to rand whole circ
    t=t(R_angle:R_angle+angle/2-1);
    x = R*sin(t)+Radius_X;
    y = R*cos(t)+Radius_Y;
    
    circ_mics_x = radius_mics*sin(t)+Radius_X;
    circ_mics_y = radius_mics*cos(t)+Radius_Y;
    [line_x,line_y]=fillline([x(1) y(1)], [x(angle/2) y(angle/2)],R*2*100);
    
%     r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(25) circ_mics_y(25) high; circ_mics_x(55) circ_mics_y(55) high; circ_mics_x(80) circ_mics_y(80) high;...
%         circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(150) circ_mics_y(150) high; circ_mics_x(180) circ_mics_y(180) high];

    r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(55) circ_mics_y(55) high;...
        circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(180) circ_mics_y(180) high];


    %% create x1 y1 x2 y2
    center=[Radius_X Radius_Y];
    start_circ_vec=[line_x(1) line_y(1)]-center;
    labels_location = 5:10:175;
    list_locations = [];
    i=1;
    while (i <= num_jumps) 
        next_speech = 1;
        number_of_loops = 0;
        while next_speech
            if number_of_loops>300
               i=1; 
               list_locations = [];
            end
            number_of_loops = number_of_loops+1;
            rand1 = randi(angle/2); 
            x1(i) = x(rand1);
            y1(i) = y(rand1);
            x1_temp(i)=x1(i);
            y1_temp(i)=y1(i);
            w=0.01*randi([1,314]);
            x1(i)=x1(i)+noise_R*sin(w);
            y1(i)=y1(i)+noise_R*cos(w);
            first_vec=[x1(i) y1(i)]-center;
            ang1=acosd((start_circ_vec(1)*first_vec(1)+start_circ_vec(2)*first_vec(2))/(norm(start_circ_vec)*norm(first_vec)));
            [~,label_first(i)] = (min(abs(labels_location - ang1)));
            
            rand2 = randi(angle/2); 
            x2(i) = x(rand2);
            y2(i) = y(rand2);
            x2_temp(i)=x2(i);
            y2_temp(i)=y2(i);
            w=0.01*randi([1,314]);
            x2(i)=x2(i)+noise_R*sin(w);
            y2(i)=y2(i)+noise_R*cos(w);
            second_vec=[x2(i) y2(i)]-center;
            ang2=acosd((start_circ_vec(1)*second_vec(1)+start_circ_vec(2)*second_vec(2))/(norm(start_circ_vec)*norm(second_vec)));
            [~,label_second(i)] = (min(abs(labels_location - ang2))); 
            if (any(list_locations == label_first(i)) || any(list_locations == label_second(i)))
                next_speech = 1;
            else
                next_speech = 0;
            end            
            
            loc_xy = [x1(i),y1(i);x2(i),y2(i)];
            dist = pdist(loc_xy,'euclidean');
            if dist<0.5
                next_speech = 1;
            end
            if i>1
                loc_xy1 = [x1(i),y1(i);x2(i-1),y2(i-1)];
                loc_xy2 = [x1(i-1),y1(i-1);x2(i),y2(i)];
                dist_last1 = pdist(loc_xy1,'euclidean');
                dist_last2 = pdist(loc_xy2,'euclidean');
                if (dist_last1<0.5 || dist_last2<0.5)
                     next_speech = 1;
                end
            end
            
        end
        list_locations=cat(1,list_locations,label_first(i));
        list_locations=cat(1,list_locations,label_second(i)); 
        i = i + 1;
    end   
    
%%  create location of the speakers 

    s_first = [x1 ; y1 ; ones(1,num_jumps)];
    s_second = [x2 ; y2 ; ones(1,num_jumps)];


    %% create location
    labels_location = 5:10:175;
    for i = 1:num_jumps
        center=[Radius_X Radius_Y];
        start_circ_vec=[line_x(1) line_y(1)]-center;
        first_vec=[x1(i) y1(i)]-center;
        second_vec=[x2(i) y2(i)]-center;
        ang1=acosd((start_circ_vec(1)*first_vec(1)+start_circ_vec(2)*first_vec(2))/(norm(start_circ_vec)*norm(first_vec)));
        ang2=acosd((start_circ_vec(1)*second_vec(1)+start_circ_vec(2)*second_vec(2))/(norm(start_circ_vec)*norm(second_vec)));
        [~,label_first(i)] = (min(abs(labels_location - ang1)));
        [~,label_second(i)] = (min(abs(labels_location - ang2)));
    end

    
    %% noise

    middle = [Radius_X Radius_Y high];
    s_noise = [Radius_X Radius_Y high];
    d_noise = norm(s_noise-middle);
    while d_noise<2
        x_noise=distance_from_woll+0.01*randi(100)*(room_x-2*distance_from_woll);
        y_noise=distance_from_woll+0.01*randi(100)*(room_y-2*distance_from_woll);
        s_noise = [x_noise y_noise high];
        d_noise = norm(s_noise-middle);
    end
    
    
    
    
    
    %% plot all
    
    figure;
    plot3(x,y,ones(1,180))
    hold on
    plot3(circ_mics_x,circ_mics_y,ones(1,180))
    hold on 
    plot3(line_x,line_y,ones(1,260));
    hold on 

    plot3([circ_mics_x(1) circ_mics_x(25) circ_mics_x(55) circ_mics_x(80) circ_mics_x(120) circ_mics_x(150) circ_mics_x(180)],...
          [circ_mics_y(1) circ_mics_y(25) circ_mics_y(55) circ_mics_y(80) circ_mics_y(120) circ_mics_y(150) circ_mics_y(180)],[1 1 1 1 1 1 1],'o')
    hold on
    
    t_noise_location = linspace(0,2*pi);
    
    for i = 1:num_jumps
        x_noise_location = noise_R*sin(t_noise_location)+x2_temp(i);
        y_noise_location = noise_R*cos(t_noise_location)+y2_temp(i);
        z_noise_location=0*t_noise_location+1;
        plot3(x_noise_location,y_noise_location,z_noise_location)
        hold on
        t_noise_location = linspace(0,2*pi);
        x_noise_location = noise_R*sin(t_noise_location)+x1_temp(i);
        y_noise_location = noise_R*cos(t_noise_location)+y1_temp(i);
        z_noise_location=0*t_noise_location+1;
        plot3(x_noise_location,y_noise_location,z_noise_location)
        
        plot3([x1(i) x2(i) x_noise],[y1(i) y2(i) y_noise],[high high high],'o')
    end
    
    hold on  
    plotcube(L, [0,0,0] ,0,[1 1 1]);   % use function plotcube 

